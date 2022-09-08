# YOLOX-CPP-ncnn

Cpp file compile of YOLOX object detection base on [ncnn](https://github.com/Tencent/ncnn).  
YOLOX is included in ncnn now, you could also try building from ncnn, it's better.

## Tutorial-Use ONNX

### Step1
Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) to build on your own device.

### Step2
Use provided tools to generate onnx file.
For example, if you want to generate onnx file of yolox-s, please run the following command:
```shell
cd <path of yolox>
python3 tools/export_onnx.py -n yolox-s
```
Then, a yolox.onnx file is generated.

### Step3
Generate ncnn param and bin file.
```shell
cd <path of ncnn>
cd build/tools/ncnn
./onnx2ncnn yolox.onnx model.param model.bin
```

Since Focus module is not supported in ncnn. Warnings like:
```shell
Unsupported slice step ! 
```
will be printed. However, don't  worry!  C++ version of Focus layer is already implemented in yolox.cpp.

### Step4
Open **model.param**, and modify it.
Before (just an example):
```
295 328
Input            images                   0 1 images
Split            splitncnn_input0         1 4 images images_splitncnn_0 images_splitncnn_1 images_splitncnn_2 images_splitncnn_3
Crop             Slice_4                  1 1 images_splitncnn_3 647 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_9                  1 1 647 652 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_14                 1 1 images_splitncnn_2 657 -23309=1,0 -23310=1,2147483647 -23311=1,1
Crop             Slice_19                 1 1 657 662 -23309=1,1 -23310=1,2147483647 -23311=1,2
Crop             Slice_24                 1 1 images_splitncnn_1 667 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_29                 1 1 667 672 -23309=1,0 -23310=1,2147483647 -23311=1,2
Crop             Slice_34                 1 1 images_splitncnn_0 677 -23309=1,1 -23310=1,2147483647 -23311=1,1
Crop             Slice_39                 1 1 677 682 -23309=1,1 -23310=1,2147483647 -23311=1,2
Concat           Concat_40                4 1 652 672 662 682 683 0=0
...
```
* Change first number for 295 to 295 - 9 = 286(since we will remove 10 layers and add 1 layers, total layers number should minus 9). 
* Then remove 10 lines of code from Split to Concat, but remember the last but 2nd number: 683.
* Add YoloV5Focus layer After Input (using previous number 683):
```
YoloV5Focus      focus                    1 1 images 683
```
After(just an example):
```
286 328
Input            images                   0 1 images
YoloV5Focus      focus                    1 1 images 683
...
```

### Step5
Use ncnn_optimize to generate new param and bin:
```shell
# suppose you are still under ncnn/build/tools/ncnn dir.
../ncnnoptimize model.param model.bin yolox.param yolox.bin 65536
```

### Step6
Copy or Move yolox.cpp file into ncnn/examples, modify the CMakeList.txt, then build yolox

### Step7
Inference image with executable file yolox, enjoy the detect result:
```shell
./yolox demo.jpg
```

## Tutorial-Use PNNX

### Step1
Clone [ncnn](https://github.com/Tencent/ncnn) first, then please following [build tutorial of ncnn](https://github.com/Tencent/ncnn/wiki/how-to-build) and
[build tutorial of pnnx](https://github.com/Tencent/ncnn/tree/master/tools/pnnx) to build on your own device.

### Step2
Use provided tools to generate torchscript file.
For example, if you want to generate onnx file of yolox-s, please run the following command:
```shell
cd <path of yolox>
python3 export_torchscript.py -n yolox-s -c yolox_s.pth
```
Then, a yolox.torchscript.pt file is generated.

### Step3
Generate ncnn param and bin file.
```shell
cd <path of ncnn>
cd build/tools/ncnn
<path to your pnnx>/pnnx YOLOX_outputs/yolox_s/yolox.torchscript.pt "inputshape=[1,3,640,640]"
```
Since Focus module is not supported in ncnn. Warnings like:
```shell
slice with step 2 is not supported
```
will be printed. However, don't  worry!  C++ version of Focus layer is already implemented in yolox-pnnx.cpp.

### Step4
Open **yolox.torchscript.ncnn.param**, and modify it.
Before (just an example):
```
7767517
224 257
Input                    in0                      0 1 in0
Split                    splitncnn_0              1 4 in0 1 2 3 4
Crop                     slice_171                1 1 1 5
Crop                     slice_170                1 1 2 6
Crop                     slice_169                1 1 3 7
Crop                     slice_168                1 1 4 8
Concat                   cat_0                    4 1 8 6 7 5 9 0=0
...
```
* Change first number for 224 to 224 - 5 = 219(since we will remove 6 layers and add 1 layers, total layers number should minus 5).
* Then remove 6 lines of code from Split to Concat, but remember the last but 2nd number: 9.
* Add YoloV5Focus layer After Input (using previous number 9):
```
YoloV5Focus      focus                    1 1 in0 9
```
After(just an example):
```
219 257
Input                    in0                      0 1 in0
YoloV5Focus              focus                    1 1 in0 9
...
```

### Step6
Copy or Move yolox-pnnx.cpp file into ncnn/examples, modify the CMakeList.txt, then build yolox

### Step7
Inference image with executable file yolox, enjoy the detect result:
## Acknowledgement

* [ncnn](https://github.com/Tencent/ncnn)
