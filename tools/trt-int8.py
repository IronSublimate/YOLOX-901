#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
from torch.utils.data import DataLoader, Dataset
from torch2trt import torch2trt

from yolox.exp import get_exp


class DatasetAdapter(Dataset):
    def __init__(self, dataset: Dataset):
        self.ds = dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        img, target, img_info, img_id = self.ds[item]
        image = img[None, ...]
        return torch.from_numpy(image).cuda()


def make_parser():
    parser = argparse.ArgumentParser("YOLOX tensorrt PTQ deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser


# @logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()

    dl = exp.get_evaluator(args.batch, False, False, False).dataloader
    ds_adapter = DatasetAdapter(dl.dataset)
    # dl_adapter = DataLoader(ds_adapter, batch_size=args.batch, num_workers=0)

    model_trt = torch2trt(
        model,
        [x],
        int8_mode=True,
        int8_calib_dataset=ds_adapter,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
