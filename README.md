# Leaping through tree space: continuous phylogenetic inference for rooted and unrooted trees

This repo hosts a minimal implementation of ```GradME```
* ```bme_jax/```: Balanced Minimum Evolution and distance-based optimisation with Phylo2Vec in Jax
* ```cfg/```: Example configuration files
* ```utils/```: Utility functions for manipulation of sequence and tree data.

## Environment setup
1. Setup the ```gradme``` environment using conda/mamba:
```
conda env create -f env.yml
```
2. Optional: if you have GPUs/TPUs, you might need to update your installation of Jax. Follow the instructions at https://github.com/google/jax
3. Install ```phangorn``` in R (4.2.2 or above):
```
install.packages("phangorn")
```

## Running GradME
1. Download the datasets (in the FASTA format) mentioned above and place them in a ```data/``` folder (e.g., in the repo)
2. Update the configuration file ```cfg/bme_config_v3.yml```, especially ```repo_path``` and ```fasta_path```
3. Run the main optimisation script: ```python -m bme_jax.main``` or use the ```demo.ipynb``` notebook


## TODOs
- Add tskit / tsinfer interface
- Compare tsinfer performance on a small phased diploid dataset w/ GradME.
- Scale!
