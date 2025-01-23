# protein-models

Exploring deep learning models for protein structure prediction. *Work in progress.*

## Overview

- Distance matrix processing and refinement using deep learning
- Structure recovery from noisy or incomplete inputs
- Latent space representations of protein structural features

The current implementation serves as a proof of concept using small (16x16) distance matrices, with plans to scale to larger and eventually full protein structures.

### Data

PDB files are downloaded and processed using the `data/download_pdbs.py` and `data/get_pairwise_matrices.py` scripts, the idea being to extract non-overlapping protein fragments and calculate pairwise distance matrices for each one.

### Sample Reconstruction

Recovery of noised protein structure using an autoencoder:

![Sample Reconstruction](data/train_denoising/figures/reconstruction_4.png)

## Next Steps

- [ ] Experiment with larger pairwise distance matrices (64x64, 128x128)
- [ ] Implement increasingly complex architectures (variational autoencoders, diffusion models,transformers, etc.)
- [ ] Render 3D structures

*A note about rendering 3D protein structures: there is a lot of complexity in making representation choices due to the nature of this particular geometric deep learning problem. Pushing 3D to the bottom of priorities for now.*

## Example Usage

```bash
# Run training
python examples/train_mlp_autoencoder.py
```
