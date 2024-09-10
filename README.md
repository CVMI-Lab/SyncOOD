<div align="center">

<h1>Can OOD Object Detectors Learn from Foundation Models?</h1>

<div>
    <a href="https://github.com/jliu-ac" target="_blank">Jiahui Liu</a>,</span>
    <a href="https://wen-xin.info" target="_blank">Xin Wen</a>,</span>
    <a href="https://openreview.net/profile?id=~Shizhen_Zhao1" target="_blank">Shizhen Zhao</a>,</span>
    <a href="https://carolchenyx.github.io/yingxianchen.github.io/" target="_blank">Yingxian Chen</a>,</span>
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
* [**Synthesize Novel Samples**](#Synthesize-Novel-Samples) for OOD object detection and more open-world tasks (comming soon).
  
* [**Train an OOD Detector**](#Train-an-OOD-detector) for achieving state-of-the-art OOD object detection (comming soon).


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
@InProceedings{liu2024can,
	author    = {Liu, Jiahui and Wen, Xin and Zhao, Shizhen and Chen, Yingxian and Qi, Xiaojuan},
	title     = {Can OOD Object Detectors Learn from Foundation Models?},
	booktitle = {European Conference on Computer Vision},
	year      = {2024}
}
```

### Acknowledgements

* This repository is based off of the work from [Du et al (ICLR 2022)](https://github.com/deeplearning-wisc/vos) and [Wilson et al (ICCV 2023)](https://github.com/SamWilso/SAFE_Official). Please support their work.
* This work is powered by [Detectron2](https://github.com/facebookresearch/detectron2), [Stable-Diffusion](https://github.com/runwayml/stable-diffusion), [ChatGPT](https://chatgpt.com/), and [Segment-Anything](https://github.com/facebookresearch/segment-anything). Thanks to these projects.


---

<span id="Synthesize-Novel-Samples"></span>
# Synthesize Novel Samples

We aim to develop an automatic, transparent, controllable, and low-cost pipeline for synthesizing scene-level images containing novel objects and provide coco-format annotations to help 1) training OOD detectors and 2) exploring more general open-world tasks (comming soon).


---

<span id="Train-an-OOD-detector"></span>
# Train an OOD Detector

We utilize synthetic Out-of-Distribution(OOD) samples and original In-Distribution(ID) samples to train a lightweight, plug-and-play OOD detector in a very efficient way, achieving state-of-the-art OOD object detection.
</br>We mainly conduct the experiments on Ubuntu 20.04 with GeForce RTX 3090 GPUs (comming soon).

