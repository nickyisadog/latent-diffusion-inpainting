# latent-diffusion-inpainting

This repository is based on [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion), with modifications for classifier conditioning and architecture improvements.

Since the original codebase is very big, complex and lack of documentation to fine-tune the original autoencoder and diffusion model. 
[How to finetune inpainting?]https://github.com/CompVis/latent-diffusion/issues/151



This repository provide made the training setup and inference easy by fixing some of the bug in the original repo.

## Requirements
If you already have the ldm environment, please skip it
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```

conda env create -f ldm/environment.yaml
conda activate ldm
```
