# EVA-7-capstone-3

# Network architecture
The proposed network consists of a fast single-stage detector YOLOv3 for box-level defect detection, and an optimized mask subnet that predicts a set of position-sensitive score maps for assembling defect masks.

<div align=center><img src="https://github.com/ZHANGKEON/DIS-YOLO/blob/master/output/architecture.jpg"/></div>

# Concrete defect [dataset]( https://drive.google.com/file/d/1UbAnTFQWShtuHlGEvYYZ4TP8tL49IM8t/view?usp=sharing)
A dataset with mask labeling of three major types of concrete surface defects: crack, spalling and exposed rebar, was prepared for training and testing of the DIS-YOLO model. In this dataset, three open-domain datasets [1-3] are exploited and merged with a bridge inspection image dataset [4] from the Highways Department. To use the dataset, you need comply with the terms and conditions of using the images from the Highways Department, therefore please write a statement and e-mail it to: czhangbd@connect.ust.hk.

# Setup for training and testing
1. Download the pretrained weights [modified Yolov3 weights]( https://drive.google.com/drive/folders/1LDE08DwQaA79-lq7NKPU7ieb_DOh9VSk?usp=sharing) and put it into the folder “pretrained_weights”.
2. Change MODEL_PATH and relevant training parameters in config.py. 
3. Create “train”, “val” and “test” folders in the “data” folder, and put the downloaded data into each folder according to the format of "train_sample". Then, run “pre_process.py” to get “ground_truth_cache.pkl” for each folder. 
4. Train model using different transfer learning and fine-tuning strategies. To fine-tune specific layers, please adjust “lock = True/False” for relevant layers in yolo3_net_pos.py.
5. After training, run “calculate_test_map.py” to evaluate mask-level mAP and mIoU.

# Test results
The instance segmentation accuracy mAP is up to 80% computed using the mask-level IoU metric of 0.5. The test speed is around 0.1 s per image (576×576 pixel resolution) on a PC running Ubuntu with an Intel® CoreTM i7-7700 CPU and a GeForce GTX 1060 6GB GPU. The comparison of test samples between different models is as follows:

<div align=center><img src="https://github.com/ZHANGKEON/DIS-YOLO/blob/master/output/sample_result.jpg"/></div>

# Training on custom dataset
To apply the code to a custom dataset, you need do the following items:
1. Prepare raw images and labeled masks for train/val/test.
2. Download the pretrained [yolov3.weights]( https://pjreddie.com/darknet/yolo/) and convert it to a “.ckpt” file. Please note that the original yolov3.weights has been trained for 80 common object classes. For custom dataset with different number of classes, you can manually crop the last class prediction channel, or select the relevant layers for restoring weight in “train_yolo3_mask.py”.
3. Change settings in config.py, such as MODEL_PATH, CLASSES and ANCHORS, etc.

# Public datasets from:
1. Li S, Zhao X and Zhou G. Automatic pixel‐level multiple damage detection of concrete structure using fully convolutional network. Comput-Aided Civ Inf Eng 2019; 34: 616–634.
2. Yang L, Li B, Li W, et al. Semantic metric 3D reconstruction for concrete inspection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, Salt Lake City, UT, 18–22 June 2018, pp.1624–1632.
3. Yang X, Li H, Yu Y, et al. Automatic pixel‐level crack detection and measurement using fully convolutional network. Comput-Aided Civ Inf Eng 2018; 33: 1090–1109.
4. Zhang C, Chang CC and Jamshidi M. Concrete bridge surface damage detection using a single‐stage detector. Comput-Aided Civ Inf Eng 2020; 35: 389–409.
# Citation
1. If you are using the dataset, please cite the papers that have contributed to the above dataset, and cite the following paper:
2. Zhang C, Chang C, Jamshidi M, Simultaneous Pixel-level Concrete Defect Detection and Grouping using a Fully Convolutional Model, Struct. Health Monit 2021.

## Demo
Visualization of Panoptic-DeepLab predictions from `demo.py`.
![Visualization of Panoptic-DeepLab](/docs/vis.png)

## Dataset
Detectron2 has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  cityscapes/
  VOC20{07,12}/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

First, prepare the Cityscapes dataset following this expected dataset structure
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```

## Backbone pre-trained weights
You probably need to use `convert-pretrain-model-to-d2.py` to convert your pre-trained backbone to the correct format first.

For Xception-65:
```
# download your pretrained model:
wget https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth -O x65.pth
# run the conversion
python convert-pretrain-model-to-d2.py x65.pth x65.pkl
```

For HRNet-48:
```
# download your pretrained model:
wget https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ -O h48.pth
# run the conversion
python convert-pretrain-model-to-d2.py h48.pth h48.pkl
```

## Panoptic-DeepLab example
Note: the only difference is we rename `train_net.py` to `train_panoptic_deeplab.py`.

### Training

To train a model with 8 GPUs run:
```bash
python train_panoptic_deeplab.py --config-file config/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --num-gpus 8
```

### Evaluation

Model evaluation can be done similarly:
```bash
python train_panoptic_deeplab.py --config-file config/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

### Benchmark network speed

If you want to benchmark the network speed without post-processing, you can run the evaluation script with `MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True`:
```bash
python train_panoptic_deeplab.py --config-file config/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True
```

### Detectron2 code structure
The decoder for Panoptic-DeepLab is defined in this file: https://github.com/facebookresearch/detectron2/blob/master/projects/Panoptic-DeepLab/panoptic_deeplab/panoptic_seg.py.  
It includes both semantic branch and instance branch.

### Cityscapes Panoptic Segmentation
Cityscapes models are trained with ImageNet pretraining.

#### Regular Conv2d in ASPP and Decoder
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="config/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">X65-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 62.6 </td>
<td align="center"> 81.5 </td>
<td align="center"> 75.7 </td>
<td align="center"> 79.4 </td>
<td align="center"> 32.8 </td>
<td align="center"><a href=https://drive.google.com/file/d/1F9Biuu9UmgfCyatP5yQTYN5V5_YyVGA2/view?usp=sharing"
">model</a></td>
</tr>
 <tr><td align="left"><a href="config/Cityscapes-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">HRNet-48</td>
<td align="center">1024&times;2048</td>
<td align="center"> 63.3 </td>
<td align="center"> 82.2 </td>
<td align="center"> 76.0 </td>
<td align="center"> 80.3 </td>
<td align="center"> 35.9 </td>
<td align="center"><a href=https://drive.google.com/file/d/1jQp844gR9NvAXmSXNuRgiY516LsRbTSN/view?usp=sharing"
">model</a></td>
</tr>
</tbody></table>

Note:
- [X65](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth): Xception-65. It is converted from TensorFlow model. You need to convert it with `convert-pretrained-model-to-d2.py` first.
- DC5 means using dilated convolution in `res5`.
- [HRNet-48](https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ): HighResolutionNet-w48. This checkpoint comes form its [original implementation](https://github.com/HRNet/HRNet-Image-Classification). You need to convert it with `convert-pretrained-model-to-d2.py` first.
- This implementation currently uses a much heavier head (with regular Conv2d) than the original paper.
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
  outputs now takes more time than the network itself.

#### DepthwiseSeparableConv2d in ASPP and Decoder
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">X65-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 61.4 </td>
<td align="center"> 81.4 </td>
<td align="center"> 74.3 </td>
<td align="center"> 79.8 </td>
<td align="center"> 32.6 </td>
<td align="center"><a href="https://drive.google.com/file/d/1ZR3YxFEdwF498NWq9ENFCEsTIiOjvMbp/view?usp=sharing
">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">HRNet-48</td>
<td align="center">1024&times;2048</td>
<td align="center"> 63.4 </td>
<td align="center"> 81.9 </td>
<td align="center"> 76.4 </td>
<td align="center"> 80.6 </td>
<td align="center"> 36.2 </td>
<td align="center"><a href="https://drive.google.com/file/d/1t1WB5GUtiwaL0UHngthX7_kWt0rBRNcO/view?usp=sharing
">model</a></td>
</tr>
</tbody></table>

Note:
- This implementation uses DepthwiseSeparableConv2d (DSConv) in ASPP and decoder, which is same as the original paper.
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
  outputs now takes more time than the network itself.

### COCO Panoptic Segmentation
COCO models are trained with ImageNet pretraining.

#### DepthwiseSeparableConv2d in ASPP and Decoder
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">Box AP</th>
<th valign="bottom">Mask AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_X_65_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">X65-DC5</td>
<td align="center">640&times;640</td>
<td align="center"> 36.7 </td>
<td align="center"> 77.4 </td>
<td align="center"> 45.8 </td>
<td align="center"> 19.9 </td>
<td align="center"> 20.5 </td>
<td align="center"><a href="https://drive.google.com/file/d/1FVj1amFkkbwL9RTba2oUYcwlD1JJMx-T/view?usp=sharing
">model</a></td>
</tr>
 <tr><td align="left"><a href="tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">HRNet-48</td>
<td align="center">640&times;640</td>
<td align="center"> 37.8 </td>
<td align="center"> 78.1 </td>
<td align="center"> 46.9 </td>
<td align="center"> 21.6 </td>
<td align="center"> 22.3 </td>
<td align="center"><a href="https://drive.google.com/file/d/17bl_n0SUXVktA0x2507dlwmti7s8FIpe/view?usp=sharing
">model</a></td>
</tr>
</tbody></table>

Note:
- **These results are trained with old COCO config files (with `MAX_SIZE_TRAIN` set to 640 instead of 960), I will try to update these numbers as soon as I have machines to train models**
- This implementation uses DepthwiseSeparableConv2d (DSConv) in ASPP and decoder, which is same as the original paper.
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
  outputs now takes more time than the network itself.
- The reproduced numbers are still lower than the original paper, this is probably due to slightly different data preprocessing.

## DeepLab example
Note: the only difference is we rename `train_net.py` to `train_deeplab.py`.

### Training
To train a model with 8 GPUs run:
```bash
python train_deeplab.py --config-file config/Cityscapes-SemanticSegmentation/deeplab_v3_plus_X_65_os16_mg124_poly_90k_bs16.yaml --num-gpus 8
```

### Evaluation
Model evaluation can be done similarly:
```bash
python train_deeplab.py --config-file config/Cityscapes-SemanticSegmentation/deeplab_v3_plus_X_65_os16_mg124_poly_90k_bs16.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint
```

## Cityscapes Semantic Segmentation
Cityscapes models are trained with ImageNet pretraining.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_X_65_os16_mg124_poly_90k_bs16.yaml">DeepLabV3+</a></td>
<td align="center">X65-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 80.1 </td>
<td align="center"><a href=https://drive.google.com/file/d/13z2R1cLEGLCIvFVFHGAianOFPi5qjpf_/view?usp=sharing"
">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_H_48_os4_poly_90k_bs16.yaml">DeepLabV3+</a></td>
<td align="center">HRNet-48</td>
<td align="center">1024&times;2048</td>
<td align="center"> 80.9 </td>
<td align="center"><a href=https://drive.google.com/file/d/1ppFGnva9WsZMMAJGx_FLZjLhvh4OBbGI/view?usp=sharing"
">model</a></td>
</tr>
</tbody></table>

Note:
- [X65](https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth): Xception-65. It is converted from TensorFlow model. You need to convert it with `convert-pretrained-model-to-d2.py` first.
- DC5 means using dilated convolution in `res5`.
- [HRNet-48](https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ): HighResolutionNet-w48. This checkpoint comes form its [original implementation](https://github.com/HRNet/HRNet-Image-Classification). You need to convert it with `convert-pretrained-model-to-d2.py` first.
