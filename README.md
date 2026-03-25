
# Code for "Holistic Physics Solver" (ICML 2025)

We provide the experimental code for both **structured meshes** (Darcy Flow, Navier-Stokes, Airfoil, Plasticity) and **irregular meshes** (Irregular Darcy Flow, Composite, Pipe Turbulence, BloodFlow, Heat Transfer).

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
- Irregular Darcy Flow, Pipe Turbulence, Composite, BloodFlow, Heat Transfer: [Google Driver](https://drive.google.com/file/d/1DjNMZjpbkJGQTxdrGQVVRKIZftOhpvGc/view?usp=sharing).
The download includes all data `.mat` files and some pre-computed LBO basis files.

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

- Composite:
```
bash ./exp_scripts/composite.sh [The Path of Composites.mat] [The Path of Composites_LBO_basis/Composites_LBO_basis.mat]
```

- Pipe Turbulence:
```
bash ./exp_scripts/pipe_turbulance.sh [The Path of Turbulence.mat]
```

- BloodFlow:
```
bash ./exp_scripts/bloodflow.sh [The Path of BloodFlow-001.mat] [The Path of BloodFlow_LBO_basis/LBO_basis.mat]
```

- Heat Transfer:
```
bash ./exp_scripts/heat_transfer.sh [The Path of HeatTransfer.mat] [The Path of HeatTransfer_LBO_basis/lbe_ev_input.mat] [The Path of HeatTransfer_LBO_basis/lbe_ev_output.mat]
```

## Project Structure

```
HPM/
├── model/
│   ├── HPM_Structured_Mesh.py    # Model for structured (regular) meshes
│   ├── HPM_Irregular_Mesh.py              # Model for irregular meshes
│   ├── HPM_Irregular_Mesh_Temporal.py     # Model for irregular meshes with temporal dimension
│   ├── HPM_Irregular_Mesh_TwoDomain.py    # Model for cross-domain irregular meshes
│   ├── Embedding.py                       # Positional and timestep embeddings
│   └── spectral_embedding/                # Spectral basis computation
├── lapy/                          # LaPy library for mesh Laplacian eigenvectors (irregular meshes)
├── utils/
│   ├── testloss.py               # Loss functions
│   └── normalizer.py             # Data normalization utilities
├── exp_airfoil.py                # Airfoil experiment (structured)
├── exp_darcy.py                  # Darcy Flow experiment (structured)
├── exp_ns.py                     # Navier-Stokes experiment (structured)
├── exp_plasticity.py             # Plasticity experiment (structured)
├── exp_irregular_darcy.py        # Irregular Darcy Flow experiment (irregular)
├── exp_composite.py              # Composite experiment (irregular)
├── exp_pipe_turbulance.py        # Pipe Turbulence experiment (irregular)
├── exp_bloodflow.py              # BloodFlow experiment (irregular, temporal)
├── exp_heat_transfer.py          # Heat Transfer experiment (irregular, two-domain)
└── exp_scripts/                  # Shell scripts for running experiments
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
  yue2025holistic,
  title={Holistic Physics Solver: Learning {PDE}s in a Unified Spectral-Physical Space},
  author={Xihang Yue and Yi Yang and Linchao Zhu},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=oB5a6yIAmF}
}
```

## Acknowledge

We thank the following open-sourced projects, which provide the basis of this work.
- https://github.com/lululxvi/deepxde
- https://github.com/neuraloperator/neuraloperator
- https://github.com/gengxiangc/NORM
- https://github.com/thuml/Transolver
- https://github.com/nmwsharp/nonmanifold-laplacian
- https://github.com/Deep-MI/LaPy
