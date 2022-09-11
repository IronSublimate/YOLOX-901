import argparse
import os
from typing import Union
from loguru import logger

import torch
from torch.utils.data import DataLoader

from yolox.exp import get_exp
from torch.quantization import quantize_dynamic_jit, quantize_jit, per_channel_dynamic_qconfig, get_default_qconfig


def make_parser():
    parser = argparse.ArgumentParser("YOLOX torchscript deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.torchscript.pt", help="output name of models"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def quantization_dynamic(traced_model: torch.nn.Module, save_path: str):
    traced_model.eval()
    qconfig_dict = {'': per_channel_dynamic_qconfig}
    quantized_model = quantize_dynamic_jit(traced_model, qconfig_dict)
    quantized_model.save(save_path)


def quantization_static(traced_model: torch.nn.Module, save_path: str, dataloader: DataLoader):
    qconfig = get_default_qconfig('fbgemm')
    qconfig_dict = {'': qconfig}

    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for sample, target in data_loader:
                model(sample)

    class DatasetAdapter(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset):
            self.ds = dataset

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, item):
            img, target, img_info, img_id = self.ds[item]
            return img, target

    ds_adapter = DatasetAdapter(dataloader.dataset)

    dl_adapter = DataLoader(ds_adapter, batch_size=dataloader.batch_size,
                            num_workers=dataloader.num_workers,
                            sampler=dataloader.sampler)
    quantized_model = quantize_jit(
        traced_model,  # TorchScript model
        qconfig_dict,  # qconfig dict
        calibrate,  # calibration function
        [dl_adapter],  # positional arguments to calibration function, typically some sample dataset
        inplace=False,  # whether to modify the model inplace or not
        debug=True)  # whether to prduce a debug friendly model or not

    quantized_model.save(save_path)


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    evaluator = exp.get_evaluator(args.batch_size, False)

    model = exp.get_model()
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    mod = torch.jit.trace(model, dummy_input)
    quantization_dynamic(mod, os.path.join(file_name, args.output_name))
    quantization_static(mod, os.path.join(file_name, args.output_name), evaluator.dataloader)
    # mod.save(os.path.join(file_name, args.output_name))
    logger.info("generated torchscript model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
