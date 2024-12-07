<div align="center">

<h1>Can OOD Object Detectors Learn from Foundation Models?</h1>

<div>
    <a href="https://github.com/jliu-ac" target="_blank">Jiahui Liu</a>,</span>
    <a href="" target="_blank">Xin Wen</a>,</span>
    <a href="https://github.com/Shizhen-ZHAO" target="_blank">Shizhen Zhao</a>,</span>
    <a href="" target="_blank">Yingxian Chen</a>,</span>
    <a href="https://xjqi.github.io/" target="_blank">Xiaojuan Qi</a><sup>&#8224</sup>  
</div>

<div>
    The University of Hong Kong&emsp;
</div>

<div>
    &#8224 corresponding author
</div>

**European Conference on Computer Vision (ECCV) 2024**

<img src="pages/figure_main.gif" width="85%"/>

</div>

* We would like to say **YES** to the title. We introduce **SyncOOD** to access open-world knowledge encapsulated within off-the-shelf foundation models by synthesizing meaningful OOD data.
* **SyncOOD** provides an ***automatic***, ***transparent***, ***controllable***, and ***low-cost*** pipeline for synthesizing **scene-level images containing novel objects with annotation boxes** via image editing.
* The **synthetic OOD samples** are filtered and employed to augment the training of **a ***lightweight***, ***plug-and-play*** OOD detector**, thus ***effectively*** optimizing the in-distribution(ID)/out-of-distribution(OOD) decision boundaries with ***minimal*** data usage.
* Explore more in the paper: [**Can OOD Object Detectors Learn from Foundation Models?**](https://arxiv.org/pdf/2409.05162) in ECCV 2024.


### Quick Guide

This repository contains code of **SyncOOD** in **two parts**:
* [**Train an OOD Detector**](#Train-an-OOD-detector) for achieving state-of-the-art OOD object detection with synthetic data:
   * [1. Environment Setup](#Train-env)
   * [2. Datasets](#Train-data)
   * [3. Base Object Detectors](#Train-base)
   * [4. Feature Extraction](#Train-feat)
   * [5. Training the OOD detector](#Train-train)
   * [6. Evaluation](#Train-inf)

* [**Synthesize Novel Samples**](#Synthesize-Novel-Samples) for OOD object detection and more open-world tasks (plan to be public).


### Abstract
Out-of-distribution (OOD) object detection is a challenging task due to the absence of open-set OOD data. Inspired by recent advancements in text-to-image generative models, such as Stable Diffusion, we study the potential of generative models trained on large-scale open-set data to synthesize OOD samples, thereby enhancing OOD object detection. We introduce SyncOOD, a simple data curation method that capitalizes on the capabilities of large foundation models to automatically extract meaningful OOD data from text-to-image generative models. This offers the model access to open-world knowledge encapsulated within off-the-shelf foundation models. The synthetic OOD samples are then employed to augment the training of a lightweight, plug-and-play OOD detector, thus effectively optimizing the in-distribution (ID)/OOD decision boundaries. Extensive experiments across multiple benchmarks demonstrate that SyncOOD significantly outperforms existing methods, establishing new state-of-the-art performance with minimal synthetic data usage.

### Key Contributions

* We investigate and unlock the potential of text-to-image generative models trained on large-scale open-set data for synthesizing OOD objects in object detection tasks.
* We introduce an automated data curation process for obtaining controllable, annotated scene-level synthetic OOD images for OOD object detection, which utilizes LLMs for novel concept discovery and visual foundation models for data annotation and filtering.
* We discover that maintaining ID/OOD image context consistency and obtaining more accurate OOD annotation bounding boxes are crucial for synthesized data to be effective in OOD object detection.
* Comprehensive experiments on multiple benchmarks demonstrate the effectiveness of our method, as we significantly outperform existing state-of-the-art approaches while using minimal synthetic data.


### Citation
If you find this work is useful, please consider citing:
```bibtex
@inproceedings{liu2025can,
  title={Can OOD Object Detectors Learn from Foundation Models?},
  author={Liu, Jiahui and Wen, Xin and Zhao, Shizhen and Chen, Yingxian and Qi, Xiaojuan},
  booktitle={European Conference on Computer Vision},
  pages={213--231},
  year={2025},
  organization={Springer}
}
```

### Acknowledgements

* This repository is based off of the work from [Du et al (ICLR 2022)](https://github.com/deeplearning-wisc/vos) and [Wilson et al (ICCV 2023)](https://github.com/SamWilso/SAFE_Official). Please support their work.
* This work is powered by [Detectron2](https://github.com/facebookresearch/detectron2), [Stable-Diffusion](https://github.com/runwayml/stable-diffusion), [ChatGPT](https://chatgpt.com/), and [Segment-Anything](https://github.com/facebookresearch/segment-anything). Thanks to these projects.


---

<span id="Train-an-OOD-detector"></span>
# Train an OOD Detector

We utilize synthetic Out-of-Distribution(OOD) samples and original In-Distribution(ID) samples to train a lightweight, plug-and-play OOD detector in a very efficient way, achieving state-of-the-art OOD object detection.
</br>We mainly conduct the experiments on Ubuntu 20.04 with GeForce RTX 3090 GPUs.

<span id="Train-env"></span>
## 1. Environment Setup

We mainly use [Conda](https://docs.conda.io/en/latest/) for installation and provide the enviroment files in ```requirements.txt``` and ```requirements.yml```, you can choose one file for setting up the environment (we use the similar environment with [Du et al (ICLR 2022)](https://github.com/deeplearning-wisc/vos) and [Wilson et al (ICCV 2023)](https://github.com/SamWilso/SAFE_Official)).

<span id="Train-data"></span>
## 2. Datasets

### Original Data

Here you should prepare:
* Two ID datasets (PASCAL-VOC, BDD-100K).
* Two OOD datasets (MS-COCO, OpenImages).

Download all the datasets (following the [Dataset Preparation](https://github.com/deeplearning-wisc/vos?tab=readme-ov-file#dataset-preparation) of the benckmark) in your own pre-defined data root path: **DATASET_DIR**. Your dataset structure should follow:
```
└── DATASET_DIR
	└── VOC_0712_converted
		|
		├── JPEGImages
		├── voc0712_train_all.json
		└── val_coco_format.json
	└── COCO
		|
		├── annotations
			├── xxx.json (the original json files)
			├── instances_val2017_ood_wrt_bdd_rm_overlap.json
			└── instances_val2017_ood_rm_overlap.json
		├── train2017
		└── val2017
	└── bdd100k
		|
		├── images
		├── val_bdd_converted.json
		└── train_bdd_converted.json
	└── OpenImages
		|
		├── coco_classes
		└── ood_classes_rm_overlap
```

### Synthetic Data
Here you should prepare two synthetic OOD datasets for training:
* SyncOOD_VOC: edited and processed from the above original dataset PASCAL-VOC.
* SyncOOD_BDD: edited and processed from the above original dataset BDD-100K.

Here you can download our processed **demo_datasets** from our [DataPage](),
</br> **or** synthesis and perpare your own synthetic OOD data with the pipeline of [Synthesize Novel Samples](#Synthesize-Novel-Samples).
</br> Your dataset structure should be updated as:
```
└── DATASET_DIR
	└── VOC_0712_converted
	└── COCO
	└── bdd100k
	└── OpenImages
 	└── SyncOOD_VOC
		|
		├── images
		└── info_raw.json
	└── SyncOOD_BDD
		|
		├── images
		└── info_raw.json
```

### Data Pre-processing

Now you should pre-process the synthetic data infomation with your pre-defined data root path **DATASET_DIR**. Ensure we are in the path (from the root path: ```SyncOOD/```) of data tools:
```bash
cd ./tools
```

Run the script with **DATASET_DIR**:
```bash
python align_ood_info.py --dataroot DATASET_DIR
```

When finishing, your dataset structure should be updated as:
```
└── DATASET_DIR
	└── VOC_0712_converted
	└── COCO
	└── bdd100k
	└── OpenImages
 	└── SyncOOD_VOC
		|
		├── images
		├── info_raw.json
		└── info.json
	└── SyncOOD_BDD
		|
		├── images
		├── info_raw.json
		└── info.json
```



<span id="Train-base"></span>
## 3. Base Object Detectors

### Detector Checkpoints

We are training a plug-and-play OOD detector with off-the-shelf base object detectors (Faster R-CNN and VOS). 
</br> Here you can follow [VOS repository](https://github.com/deeplearning-wisc/vos) to train your own base object detectors,
</br> **or** download our well-trained **base_detectors** checkpoints from our [DataPage]().

Save all the checkpoints in a flexible detector root path: **DETECTOR_DIR** and follow the structure as:
```
└── DETECTOR_DIR
	└── frcnn_voc.pth
	└── vos_voc.pth
	└── frcnn_bdd.pth
	└── vos_bdd.pth
```

### Detector Configs

Ensure we are in the path (from the root path: ```SyncOOD/```) of base detectors:
```bash
cd ./detection/configs
```
Here please ensure which base object detector you would like to use (**Faster R-CNN** or **VOS**):

* For **VOC** as ID dataset:
</br> Modify the path of ```WEIGHTS``` in ```line4``` in ```VOC-Detection/faster-rcnn/vanilla.yaml``` as:
</br>```DETECTOR_DIR/frcnn_voc.pth``` for Faster R-CNN or ```DETECTOR_DIR/vos_voc.pth``` for VOS.

* For **BDD** as ID dataset: 
</br> Modify the path of ```WEIGHTS``` in ```line4``` in ```BDD-Detection/faster-rcnn/vanilla.yaml``` as:
</br>```DETECTOR_DIR/frcnn_bdd.pth``` for Faster R-CNN or ```DETECTOR_DIR/vos_bdd.pth``` for VOS.



<span id="Train-feat"></span>
## 4. Feature Extraction

Feature extraction may consume a lot of disk space and memory, especially on the BDD100K dataset.
</br> If you are using the checkpoints provided from our **base_detectors**, here you can download our **extracted_features** from our [DataPage]() into the [*updated dataset structure*](#Train-feat-data) and skip this step,
</br> **or** follow the instructions to extract your own features:

Ensure we are in the path (from the root path: ```SyncOOD/```) of feature extraction:
```bash
cd ./OOD_OBJ_DET
```

**Firstly** we extract **ID features** from original ID samples:
```bash
sh feature_extraction_id.sh
```
- Set ```CUDA_VISIBLE_DEVICES``` with a GPU ID number (e.g. ```CUDA_VISIBLE_DEVICES=0```);
- Set ```--tdset``` with the ID dataset (```--tdset VOC``` or ```--tdset BDD```);
- Set ```--dataset-dir``` as ```--dataset-dir DATASET_DIR``` with your pre-defined data root path **DATASET_DIR**.

**Then** we extract **OOD features** from synthetic OOD samples:
```bash
sh feature_extraction_ood.sh
```
- Set ```CUDA_VISIBLE_DEVICES``` with a GPU ID number (e.g. ```CUDA_VISIBLE_DEVICES=0```);
- Set ```--tdset``` with the related ID dataset (```--tdset VOC``` or ```--tdset BDD```);
- Set ```--dataset-dir``` as ```--dataset-dir DATASET_DIR``` with your pre-defined data root path **DATASET_DIR**.



<span id="Train-feat-data"></span>
**Finally** your *updated dataset structure* should be:
```
└── DATASET_DIR
	└── VOC_0712_converted
	└── COCO
	└── bdd100k
	└── OpenImages
 	└── SyncOOD_VOC
	└── SyncOOD_BDD
	└── VOC_features
		|
		├── VOC-RCNN-RN50-id.hdf5
		└── VOC-RCNN-RN50-ood.hdf5
	└── BDD_features
		|
		├── BDD-RCNN-RN50-id.hdf5
		└── BDD-RCNN-RN50-ood.hdf5
```



<span id="Train-train"></span>
## 5. Training the OOD detector

Ensure we are in the path (from the root path: ```SyncOOD/```) of OOD detector training:
```bash
cd ./OOD_OBJ_DET
```

Then train an OOD detector:
```bash
sh train.sh
```
- Set ```CUDA_VISIBLE_DEVICES``` with a GPU ID number (e.g. ```CUDA_VISIBLE_DEVICES=0```);
- Set ```--tdset``` with the ID dataset (```--tdset VOC``` or ```--tdset BDD```);
- Set ```--dataset-dir``` as ```--dataset-dir DATASET_DIR``` with your pre-defined data root path **DATASET_DIR**.

The obtained OOD detector checkpoints are saved together with the extracted features, so the current data structure is:
```
└── DATASET_DIR
	└── VOC_0712_converted
	└── COCO
	└── bdd100k
	└── OpenImages
 	└── SyncOOD_VOC
	└── SyncOOD_BDD
	└── VOC_features
		|
		├── VOC-RCNN-RN50-id.hdf5
		├── VOC-RCNN-RN50-ood.hdf5
		└── VOC-RCNN-RN50-mlp.pth
	└── BDD_features
		|
		├── BDD-RCNN-RN50-id.hdf5
		├── BDD-RCNN-RN50-ood.hdf5
		└── BDD-RCNN-RN50-mlp.pth
```

<span id="Train-inf"></span>
## 6. Evaluation
Ensure we are in the path (from the root path: ```SyncOOD/```) of evaluating the obtained OOD detectors:
```bash
sh evaluation.sh
```
- Set ```CUDA_VISIBLE_DEVICES``` with a GPU ID number (e.g. ```CUDA_VISIBLE_DEVICES=0```);
- Set ```--tdset``` with the ID dataset (```--tdset VOC``` or ```--tdset BDD```);
- Set ```--dataset-dir``` as ```--dataset-dir DATASET_DIR``` with you pre-defined data root path **DATASET_DIR**.
- Set ```--mlp-path``` as ```--mlp-path DATASET_DIR/xxx_features/xxx-RCNN-RN50-mlp.pth``` with your pre-defined data root path **DATASET_DIR** and replace ```xxx``` with your ID dataset (```VOC``` or ```BDD```).

Finally you can get FPR95, AUROC, and AUPR of your OOD detector on two OOD dataset.


---

<span id="Synthesize-Novel-Samples"></span>
# Synthesize Novel Samples

We aim to develop an automatic, transparent, controllable, and low-cost pipeline for synthesizing scene-level images containing novel objects and provide coco-format annotations to help 1) training OOD detectors and 2) exploring more general open-world tasks (comming soon).


