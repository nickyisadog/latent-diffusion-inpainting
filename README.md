# latent-diffusion-inpainting

This repository is based on [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), with modifications for classifier conditioning and architecture improvements.

Since the original codebase is very big, complex and lack of documentation to fine-tune the original autoencoder and diffusion model.

It is really diffcult to fine tune existing pre trained model to produce good result.


#### Iusses in the original repository

[How to finetune inpainting? #151](https://github.com/CompVis/latent-diffusion/issues/151)

[how to train Inpainting model using our own datasets? #280](https://github.com/CompVis/latent-diffusion/issues/280)

[Details about training inpainting model #9](https://github.com/CompVis/latent-diffusion/issues/9)

[how to train inpainting model with my own datasets #265](https://github.com/CompVis/latent-diffusion/issues/265)

[Training inpainting model on new datasets #298](https://github.com/CompVis/latent-diffusion/issues/298)

[Reproduction problem while training inpainting model #159](https://github.com/CompVis/latent-diffusion/issues/159)


#### Hardware requirement
Without pretraining, it would take 8 V100 GPUs to produce satisfactory result. 

With finetuning, 1 3090 is enough for transfer learning to medical images( in my case )

This repository provide made the fine tuning setup and inference easy by fixing some of the bug in the original repo.

#### Major Changes

1. Load and Fine tune autoencoder (Very important for transfer learning )
2. Load and fine tune latent diffusion model
3. Combine trained autoencoder with latent diffusion model
4. Inference example for both model
5. Support easy data and mask loading
6. Fixed some bug when training inpainting model


#### Result
Original Image
![rdm-figure](assets/original_image.png)

One polyp
![rdm-figure](assets/1.gif)


Two polyp

![rdm-figure](assets/2.gif)

## Requirements
If you already have the ldm environment, please skip it

A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```

conda env create -f ldm/environment.yaml
conda activate ldm
```
## Finetune your Latent diffusion model
First, prepare the images and masks with the same format as in kvasir-seg folder

Second, modify the data path in config.yaml( it should be in ldm/models/ldm/inpainting_big/config.yaml )

Then run the following command
```
CUDA_VISIBLE_DEVICES=0 python main.py --base ldm/models/ldm/inpainting_big/config.yaml --resume /ldm/models/ldm/inpainting_big/last.ckpt --stage 1 -t --gpus 0,

```
