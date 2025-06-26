<p align="center">
  <img src="./assets/profound_logo.png" alt="ProFound Logo" width="400"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"/>
</p>

# ProFound: Vision Foundation Models for Prostate Multiparametric MR Images
ProFound is a collection of vision foundation models, pre-trained on multiparametric 3D magnetic resonance images from thousands of prostate cancer patients. 

We aim to open-source code for pre-training, fine-tuning, and evaluation, together with weights of pre-trained and fine-tuned ProFound models. This is an ongoing effort, so please check back later for updates.


## Downstream tasks
Profound can be fine-tuned for a wide range of prostate imaging tasks. Switch to the `demo` branch for examples:
```batch
git checkout demo
```

### Cancer classification
- Download weights: [fine-tuned weights](https://your-download-link-here.com)
- Run:
  ```bash
  python ./demo/classification_pirads.py
  ```
- Example output:  
  ![Cancer segmentation example](./assets/cancer_segmentation_example.png)


### Gland segmentation
- Download weights: [fine-tuned weights](https://your-download-link-here.com)
- Run:
  ```bash
  python ./demo/segmentation_gland.py
  ```
- Example output:  
  ![Gland segmentation example](./assets/gland_segmentation_example.png)

*More tasks are on the way...*

## Pre-trained models

### Available models
- **ProFound-alpha**: [Download pre-trained weights](https://your-download-link-here.com)
  > Pre-trained on approximately 5,000 international, cross-institute, multiparametric prostate MRI studies, each of which includes T2w, ADC and high-b DWI volumes

*More models coming soon!*
