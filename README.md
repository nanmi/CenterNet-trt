# CenterNet deploy with TensorRT :honeybee:

This project base on https://github.com/xingyizhou/centernet

## System Requirements

cuda 11.4

TensorRT 8+

OpenCV 4.0+(you can also use light opencv for mini version of nihui [https://github.com/nihui/opencv-mobile])

## 1. download CenterNet project
```shell
git clone https://github.com/xingyizhou/centernet

# and download checkpoint file
# such as ctdet_coco_dla_2x.pth
```

## 2. update DCNv2 pytorch plugin with my plugin 
copy `DCNv2.py` in path `CenterNet/src/lib/models/networks/`
and modify all `import DCN` to `from DCNv2 import DCN`, this DCNv2 has some changes
* core implement use torchvision.ops.deform_conv
* use split op instead of chunk op
```python
        # 1. method 1
        # o1, o2, mask = torch.chunk(out, 3, dim=1)
        # 2. method 2
        [o1, o2, mask] = torch.split(out, int(out.shape[1]/3), dim=1)
```

## 3. test pytorch run and result
```
cd CenterNet/src/
python demo.py ctdet --demo ../images/16004479832_a748d55f21_k.jpg --load_model ctdet_coco_dla_2x.pth
```

## 4. export onnx for TensorRT
use my export onnx script
move `export_onnx.py` script to `CenterNet/src/`
```shell
python export_onnx.py ctdet
```

note that
* this export script do not export heatmap sigmoid operator and decode's maxpooling operator !!! because postprocess will run sigmoid operator, check onnx it


## 5. modify onnx custom op PythonOp to DCNv2_TRT
use python script `modify_dcn_plugin.py`
```shell
python modify_dcn_plugin.py -i {your export onnx model in step 4}.onnx -o {model modified onnx}.onnx
```

## 6. build dcnv2 plugin for build engine
```shell
cd dcnv2_trt
mkdir build && cd build
cmake ..
make -j
```
and you will get a plugin library named `libdcnv2.so`

## 7. build your TensorRT engine with libdcnv2.so
```shell
trtexec --onnx={model modified onnx}.onnx --workspace=10240 --saveEngine=ctdet_fp32.engine --plugins=dcnv2_trt/build/libdcnv2.so
```
and you will get TensorRT engine

## 8. build test project of TensorRT
```shell
cd centernet-trt
mkdir build && cd build
cmake ..
make -j
```

## 9. run test
```shell
./ctdet ../ctdet_fp32.engine -i ../test.jpg
```

## TODO
* pose infer with TensorRT
