
# Code for HPM

We provide the experimental code for both **structured meshes** (Darcy Flow, Navier-Stokes, Airfoil, Plasticity) and **irregular meshes** (Irregular Darcy Flow).

## Environment Installation

Create and activate an Anaconda Environment:
```
conda create -n HPM python=3.8
conda activate HPM
```

Install required packages with the following commands:
```
pip install -r requirement.txt
```

## Data Preparation

Download the dataset from the following links, and then unzip them in a specific directory.

### Structured Mesh
- Darcy Flow: [Google Driver](https://drive.google.com/file/d/1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf/view?usp=sharing)
- Navier-Stokes: [Google Driver](https://drive.google.com/file/d/1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5/view?usp=sharing)
- Airfoil: [Google Driver](https://drive.google.com/drive/folders/1JUkPbx0-lgjFHPURH_kp1uqjfRn3aw9-?usp=sharing)
- Plasticity: [Google Driver](https://drive.google.com/file/d/14CPGK_ljae5c6dm2nRraY2kIDt39JX3d/view?usp=sharing)

### Irregular Mesh
- Irregular Darcy Flow: 
- 

## Experiment Running

### Structured Mesh Experiments

- Darcy Flow:
```
bash ./exp_scripts/darcy.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/Darcy
```

- Navier-Stokes:
```
bash ./exp_scripts/ns.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/Navier-Stokes
```

- Airfoil:
```
bash ./exp_scripts/airfoil.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/naca
```

- Plasticity:
```
bash ./exp_scripts/plasticity.sh [The Directory of Downloaded Data]
# The Provided Directory should be like: XXX/plasticity
```

### Irregular Mesh Experiments

- Irregular Darcy Flow:
```
bash ./exp_scripts/irregular_darcy.sh [The Path of Darcy.mat]
# The Provided Path should be like: XXX/Darcy.mat
```

## Project Structure

```
HPM/
├── model/
│   ├── HPM_Structured_Mesh.py    # Model for structured (regular) meshes
│   ├── HPM_Irregular_Mesh.py     # Model for irregular meshes
│   ├── Embedding.py              # Positional and timestep embeddings
│   └── spectral_embedding/       # Spectral basis computation for structured meshes
├── lapy/                          # LaPy library for mesh Laplacian eigenvectors (irregular meshes)
├── utils/
│   ├── testloss.py               # Loss functions
│   └── normalizer.py             # Data normalization utilities
├── exp_airfoil.py                 # Airfoil experiment (structured)
├── exp_darcy.py                   # Darcy Flow experiment (structured)
├── exp_ns.py                      # Navier-Stokes experiment (structured)
├── exp_plasticity.py              # Plasticity experiment (structured)
├── exp_irregular_darcy.py         # Irregular Darcy Flow experiment (irregular)
└── exp_scripts/                   # Shell scripts for running experiments
```

## Acknowledge

We thank the following open-sourced projects, which provide the basis of this work.
- https://github.com/neuraloperator/neuraloperator
- https://github.com/gengxiangc/NORM
- https://github.com/thuml/Transolver
- https://github.com/nmwsharp/nonmanifold-laplacian
- https://github.com/Deep-MI/LaPy
