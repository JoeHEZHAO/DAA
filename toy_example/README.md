# Toy Example for Regularized Gumbel-GAN

## Included Content:
* Discrete Gaussian Grid Dataset, modified from continuous gaussin grid dataloader;
* utils.py & visualizer.py for visual demo;
* NDiv from CVPR2019 [NDiv](https://github.com/B1ueber2y/NDiv);
* Three ipynb files:
    - GCGAN_t=1.ipynb shows mode-collapse of Gumbel-Conditional GAN for temperature=1
    - GCGAN_t=1000.ipynb, shows over-diversification for temperature=1000
    - GCGAN_Regularized_t=1.ipynb, shows regularized results that captures valid modes but **not perfectly**;

## Assume Cuda GPU Support. 