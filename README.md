# ProFound
ProFound is a collection of vision foundation models, pre-trained on multi-parametric 3D magnetic resonance images from thousands of prostate cancer patients. 

We aim to open-source code for pre-training, fine-tuning, and evaluation, together with weights of pre-trained and fine-tuned models. This is an ongoing effort, so please check back later for updates.


## Downstream tasks
Profound can be fine-tuned for a wide range of prostate cancer imaging tasks. Switch to the `demo` branch for examples:
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


## Pre-trained models

### Available models
- **ProFound-alpha**: [Download pre-trained weights](https://your-download-link-here.com)
  *Pre-trained on approximately 5,000 multi-parametric prostate MRI studies.*
  
