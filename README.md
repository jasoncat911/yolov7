# Quantization YOLOv7

Implementation of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="./figure/performance.png" width="79%"/>
    </a>
</div>

## Web Demo

- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/akhaliq/yolov7) using Gradio. Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/yolov7)

## Performance 

MS COCO

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps | batch 32 average time |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv7**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) | 640 | **51.4%** | **69.7%** | **55.9%** | 161 *fps* | 2.8 *ms* |
| [**YOLOv7-X**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) | 640 | **53.1%** | **71.2%** | **57.8%** | 114 *fps* | 4.3 *ms* |
|  |  |  |  |  |  |  |
| [**YOLOv7-W6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) | 1280 | **54.9%** | **72.6%** | **60.1%** | 84 *fps* | 7.6 *ms* |
| [**YOLOv7-E6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) | 1280 | **56.0%** | **73.5%** | **61.2%** | 56 *fps* | 12.3 *ms* |
| [**YOLOv7-D6**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) | 1280 | **56.6%** | **74.0%** | **61.8%** | 44 *fps* | 15.0 *ms* |
| [**YOLOv7-E6E**](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt) | 1280 | **56.8%** | **74.4%** | **62.1%** | 36 *fps* | 18.7 *ms* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>


### create the Vitis AI 3.0 docker container, please follow instruction in the Vitis-AI procedure.
### link to [repo](https://github.com/jasoncat911/Vitis-AI)

</details>

## Quantization(PTQ)

Following command to be excuted in Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>


### Inspect float model
``` shell
python quantization.py --model yolov7/cfg/deploy/yolov7.yaml --weights yolov7/cfg/deploy/yolov7.pt --output-dir yolov7/ --quant_mode 'float' --inspect
```

### Quantize model using coco eval dataset(~5000)
``` shell
python quantization.py --model yolov7/cfg/deploy/yolov7.yaml --weights yolov7/cfg/deploy/yolov7.pt --output-dir yolov7/ --quant_mode 'calib' --batch size_fit_your_machine
```

### Test quantized model using coco eval dataset(~5000)
``` shell
python quantization.py --model yolov7/cfg/deploy/yolov7.yaml --weights yolov7/cfg/deploy/yolov7.pt --output-dir yolov7/ --quant_mode 'test' --batch size_fit_your_machine
```
### Export quantized model(.xmodel)
``` shell
python quantization.py --model yolov7/cfg/deploy/yolov7.yaml --weights yolov7/cfg/deploy/yolov7.pt --output-dir yolov7/ --quant_mode 'test' --batch 1 --deploy
```

### Compile model(XIR based)
``` shell
vai_c_xir -x yolov7/quantize_result/Model_int.xmodel -a /path_to_DPU_TRD/arch.json -o yolov7/quantize_result/ -n output_net_name
```
</details>


## Testing float model

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-e6e.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt)

``` shell
python quantization.py --model yolov7/cfg/deploy/yolov7.yaml --weights yolov7/cfg/deploy/yolov7.pt --output-dir yolov7/ --quant_mode 'float' --batch size_fit_your_machine

```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51206
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.69730
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.55521
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.35247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55937
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.63765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.68772
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.53766
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.73549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.83868
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`

## Training

go to main branch for original yolov7 training 

## Transfer learning

go to main branch for original yolov7 transfer learning 

## Re-parameterization
model in from deploy folder is already after re-parameterization 
See [reparameterization.ipynb](tools/reparameterization.ipynb)

## Inference

Embedded inference comming soon 

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="59%"/>
    </a>
</div>




## Instance segmentation
### qunatization model does not include mask head 
[`code`](https://github.com/WongKinYiu/yolov7/tree/mask) [`yolov7-mask.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-mask.pt)

See [instance.ipynb](https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb).

<div align="center">
    <a href="./">
        <img src="./figure/mask.png" width="59%"/>
    </a>
</div>

## Instance segmentation

[`code`](https://github.com/WongKinYiu/yolov7/tree/u7/seg) [`yolov7-seg.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-seg.pt)

YOLOv7 for instance segmentation (YOLOR + YOLOv5 + YOLACT)

| Model | Test Size | AP<sup>box</sup> | AP<sub>50</sub><sup>box</sup> | AP<sub>75</sub><sup>box</sup> | AP<sup>mask</sup> | AP<sub>50</sub><sup>mask</sup> | AP<sub>75</sub><sup>mask</sup> |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **YOLOv7-seg** | 640 | **51.4%** | **69.4%** | **55.8%** | **41.5%** | **65.5%** | **43.7%** |

## Anchor free detection head

[`code`](https://github.com/WongKinYiu/yolov7/tree/u6) [`yolov7-u6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-u6.pt)

YOLOv7 with decoupled TAL head (YOLOR + YOLOv5 + YOLOv6)

| Model | Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> |
| :-- | :-: | :-: | :-: | :-: |
| **YOLOv7-u6** | 640 | **52.6%** | **69.7%** | **57.3%** |


## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
