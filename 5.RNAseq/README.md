## Overview

This folder contains scripts and data for training and optimizing ElasticNet models to predict the latent representations of multi-gene processes, found earlier, from the DepMap RNAseq data. The workflow involves:

Loading the preexisting latent dimension dataframes

Using DepMap RNAseq data as input features

Training and optimizing ElasticNet models to predict each latent dimension individually

Visualizing predicted vs. actual latent values for the samples in the test set for each dimension

Enhancing visualization by adding drug scores as point sizes and cancer types as color for the top drug correlated with each latent dimension