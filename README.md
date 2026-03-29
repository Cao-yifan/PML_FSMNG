# Partial Multi-label Feature Selection via Entropy-weighted Multi-scale Neighborhood Granular Label Distribution Learning


This repository contains the official MATLAB implementation for the paper: **"Partial Multi-label Feature Selection via Entropy-weighted Multi-scale Neighborhood Granular Label Distribution Learning" (PML-FSMNG)**.



## Abstract

Partial multi-label feature selection aims to identify discriminative features from data where each instance is associated with an ambiguous candidate label set. Existing methods are typically built upon single-scale modeling assumptions and may fail to fully exploit the multi-granularity structure underlying instance–label relationships. To address this limitation, we propose a novel framework termed **PML-FSMNG**, which integrates entropy-weighted multi-scale neighborhood granules with label distribution learning. 

Specifically, multi-scale neighborhood systems are constructed to estimate label distinguishability at multiple structural scales, and Shannon entropy is employed to adaptively fuse scale-specific label distributions into a robust soft supervisory signal. Based on the learned label distribution, an embedded sparse regression model with $\ell_{2,1}$-norm regularization is developed for discriminative feature selection, together with an entropy-regularized adaptive graph learning mechanism to preserve intrinsic geometric structure. Extensive experiments on benchmark datasets demonstrate that the proposed method consistently outperforms several state-of-the-art approaches, validating the effectiveness of multi-scale modeling and entropy-guided adaptive learning under label ambiguity.

## Requirements

To run this code, you need:
* **MATLAB** (Tested on R2020a and above)

## Repository Structure

* `main.m`: The main entry script. It loads the dataset, performs cross-validation, executes the feature selection pipeline, and prints comprehensive evaluation metrics.
* `PML_FSMNG.m`: The core optimization solver. It iteratively optimizes the objective function with $\ell_{2,1}$-norm sparsity and graph regularization.
* `multiscale_entropy_weighted_labels.m`: Generates the multi-scale neighborhood soft labels and calculates the entropy-driven adaptive weights.
* `getPartialLabel.m`: Helper function to generate partial/ambiguous labels for experimental settings.
* `music_emotion.mat`: A sample benchmark dataset provided for demonstration purposes.
Please get in touch with me if you have any questions about running this code!
cao_yifan@buaa.edu.cn

