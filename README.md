<p align="center">
  <img src="./assets/profound_logo.png" alt="ProFound Logo" width="400"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch 2.6"/>
  <img src="https://img.shields.io/badge/MONAI-1.5-005BBB?style=flat&logo=monai&logoColor=white" alt="MONAI 1.5"/>
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?style=flat&logo=nvidia&logoColor=white" alt="CUDA 11.8"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"/>
  <img src="https://img.shields.io/badge/HuggingFace-Demo-orange?logo=huggingface&logoColor=white" alt="Hugging Face Demo"/>
</p>



# ProFound: Vision Foundation Models for Prostate Multiparametric MR Images
ProFound is a suite of vision foundation models, pre-trained on multiparametric 3D magnetic resonance (MR) images from large collections of prostate cancer patients. 

We aim to open-source all code for pre-training, fine-tuning, and evaluation, together with weights of pre-trained and fine-tuned ProFound models. This is an ongoing effort, so please check back later for updates.


## 🤗 Try ProFound Online

Interact with ProFound directly in your browser via [our Hugging Face Space](https://huggingface.co/spaces/wxyi088/ProFound).


## 🐣 Downstream Clinical Tasks
Profound can be fine-tuned for a wide range of prostate imaging tasks. Switch to the `demo` branch for examples:
```batch
git checkout demo
```

- Download weights and example images [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmapyw0_ucl_ac_uk/ElyR-Bc7QqVAjhShIptm9K8BJsSb6QKKqJn0XolSEj0vgQ?e=MsrMCf).

- Decompress (if needed) and place the downloaded folders, `checkpoints` and `demo`, at the repository root directory.

- Configure dependencies.
  - Install PyTorch version specified in `requirements-pytorch.txt`.
  - Install MONAI and other packages:
   ```bash
   pip install -r requirements.txt
   ```

- Run the following tasks:


### Radiological cancer classification
- **Run**:
  ```bash
  sh demo_run_classification.sh
  ```
<!-- - **Example output:**  
  ![Cancer segmentation example](./assets/cancer_segmentation_example.png) -->

### Lesion segmentation
- **Run**:
  ```bash
  sh demo_run_lesion_segmentation.sh
  ```
<!-- - **Example output:**  
  ![Gland segmentation example](./assets/anatomy_segmentation_example.png) -->

### Anatomy segmentation
- **Run**:
  ```bash
  sh demo_run_anatomy_segmentation.sh
  ```

<!-- ### Cancer localisation
- **Download weights**: [fine-tuned weights](https://your-download-link-here.com)
- **Run**:
  ```bash
  python ./demo/localisation_pirads3.py
  ```
- **Example output**:  
  ![Gland segmentation example](./assets/localisation_pirads3_example.png) -->

*More tasks are on the way...*



## 🥚 Pre-trained Models

### Available models
- **ProFound-alpha**: [Download pre-trained weights](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmapyhu_ucl_ac_uk/Et-apmeR2JRMsWRPV0VbQ40BZETdEUmWeq40ulwucYoW-A?e=T6y7F0)
> Pre-trained on approximately 5,000 international, cross-institute, multiparametric prostate MRI studies, each of which includes T2w, ADC and high-b DWI volumes

*More models coming soon!*



## 🤝 Contact
Open an issue for questions and feedback.




[//]: < ## 🌞 Acknowledgement>
[//]: <This work is supported by the International Alliance for Cancer Early Detection, an alliance between Cancer Research UK, Canary Center at Stanford University, the University of Cambridge, OHSU Knight Cancer Institute, University College London and the University of Manchester.>

[//]: <This work is also supported by the National Institute for Health Research University College London Hospitals Biomedical Research Centre.>

[//]: <The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023].>

---
