# CryoNet.Refine

<div align="center">

**AI-Driven Real-Space Refinement for Cryo-EM Structures**

[![Paper](https://img.shields.io/badge/Paper-ICLR%202026-blue)](https://openreview.net/forum?id=NwzY2yhlme)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview


**CryoNet.Refine**, a new AI-driven real-space refinement framework for proteins, DNA, RNA, and their complexes. CryoNet.Refine employs a **one-step diffusion model** that tightly integrates experimental density information with stereochemical and physical constraints, enabling rapid convergence to structures that simultaneously exhibit high model–map correlation and model geometric metrics. Across comprehensive benchmarks, CryoNet.Refine delivers **substantial improvements** over traditional refinement methods in both model–map correlation and model geometric metrics, while achieving significantly lower computational cost. By providing an automated, scalable, and robust refinement pipeline, CryoNet.Refine represents an important step toward next-generation cryo-EM model refinement and broadens the applicability of AI-assisted structural biology.

---

## 🎯 Key Features

- ⚡ **Fast Convergence**: One-step diffusion model enables rapid refinement
- 🎯 **High Accuracy**: Superior model–map correlation and geometric metrics compared to Phenix.real_space_refine
- 🔧 **Automated Pipeline**: Eliminates the need for extensive manual tuning
- 💰 **Cost-Effective**: Significantly lower computational cost than traditional refinement pipelines
- 🧬 **Universal**: Works with proteins, DNA, RNA, and their complexes

---

## 🖼️ Framework Overview

### Model Architecture
<div align="center">
  <img src="https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/cryonet.refine_framework.png" alt="CryoNet.Refine Framework" width="90%">
</div>

### Refinement Process
<div align="center">
  <img src="https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/cryonet.refine.gif" alt="Refinement Process" width="90%">
</div>

---

## 🌐 Web Server

Try CryoNet.Refine online without installation:

**Web Server**: [https://cryonet.ai/refine/](https://cryonet.ai/refine/)


---

## 🚀 Installation

### 1.1 Create Conda Environment

From the project root (`cryonet.refine`):

```bash
conda env create -f cryonet.refine_env.yml
conda activate cryonet.refine
```

### 1.2 Install PyTorch (CUDA 12.1)

Use the official PyTorch CUDA 12.1 wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For users in China, you can alternatively use the Aliyun mirror:

```bash
pip install torch==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu12
```

### 1.3 Install cctbx chem_data

CryoNet.Refine uses cctbx geometry libraries and requires `chem_data`:

```bash
wget https://github.com/cctbx/cctbx_project/releases/download/v2025.10/chem_data-2025.10-pyhe0d8492_0.conda
conda install ./chem_data-2025.10-pyhe0d8492_0.conda
```

Make sure this is done inside the `cryonet.refine` conda environment.

---

## 💻 Usage

### Quick Start

From the project root (`cryonet.refine`), run:

```bash
sh ./run.sh ./examples/0775_af3.cif ./examples/emd_0775.map 3.6 ./output
```

### Arguments

- `./examples/0775_af3.cif` – Input model (PDB/mmCIF) to be refined path. (Download the AlphaFold 3 + matchmaker structure model from [0775_af3.cif](https://cryonet.oss-cn-beijing.aliyuncs.com/cryonet.refine/0775_af3.cif))
- `./examples/emd_0775.map` – Input cryo-EM density map path. (Download the density map file from [EMD-0775](https://www.ebi.ac.uk/emdb/EMD-0775))
- `3.6` – Map resolution (Å)
- `./output` – Output directory where refined models and logs will be written

The script will run the full refinement pipeline and save refined atomic model into `./output`.

> ⚠️ **Large density map origin (`--ignore_origin`)**  
> A very large map origin can hurt refinement. With `--ignore_origin`, the density and atomic coordinates are shifted together so they are aligned near the origin(0,0,0) during refinement, which removes that offset effect. Uncomment `--ignore_origin` in `run.sh` when needed.

### Validation Configuration in `run.sh`

- `run.sh` currently defines:
  - `phenix_env="/opt/phenix-1.21.1-5286/phenix_env.sh"`
  - `chimerax_cmd="/usr/bin/chimerax"`
- You can modify these two variables in `run.sh` to match your local environment.
- Validation is optional and disabled by default in `run.sh`.  
  To enable it, uncomment `--validate_output` in the `run.sh`.
- When validation is enabled, validation outputs are written next to each validated CIF file (same directory), including `.vcx`.

### Run Validation Script Directly

You can also run validation directly (without `run.sh`):

```bash
python  ./CryoNetRefine/data/output/metrics_validation.py \
  <pdb_file> <map_file> -r <resolution> \
  --phenix_env /opt/phenix-1.21.1-5286/phenix_env.sh \
  --chimerax_cmd /usr/bin/chimerax
```

This command also writes validation files (`.vcx`) in the same directory as the target CIF/PDB file.

---

## 📁 Project Structure

```
cryonet.refine/
├── CryoNetRefine/          # Core library
│   ├── libs/               # Geometry, protein representation, density utilities
│   ├── model/              # Network architecture and refinement engine
│   └── data/               # Feature generation, IO, and preprocessing
├── params/                 # Model checkpoints (auto-downloaded if missing)
├── run.sh                  # Convenience script for running refinement
├── main.py                 # Main refinement script
└── cryonet.refine_env.yml  # Conda environment specification
```

---

## ⚠️ Limitations

Despite its effectiveness, CryoNet.Refine has several limitations:

### Supported macromolecules
The current model supports proteins, DNA, RNA, and their complexes.

### Unsupported ligands and covalent modifications
CryoNet.Refine does not support small-molecule ligands, or covalent modifications. These entities require specialized modeling strategies beyond the current framework.

### Performance Degradation with missing residues
Refinement performance may decrease for structures with many missing or unresolved residues, as incomplete backbone connectivity limits the effectiveness of density-guided geometric constraints.

---

## 🙏 Acknowledgements

CryoNet.Refine borrows and adapts code from the **Boltz** project  
(https://github.com/jwohlwend/boltz.git). Parts of the original codebase were
extended and modified, and the model was further fine-tuned for our specific
refinement setting. We gratefully acknowledge the Boltz authors for making their
work openly available.

This work also makes use of the *Computational Crystallography Toolbox* (cctbx)
project (https://github.com/cctbx/cctbx_project), from which we utilize geometry
restraint interfaces, including bond, angle, and non-bonded interaction terms.
We thank the cctbx developers for providing a robust and well-maintained framework
for macromolecular geometry modeling and refinement.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



---

## 📚 Citation

If you use CryoNet.Refine in your research, please cite our paper:

```bibtex
@inproceedings{huang2026cryonet,
  title={CryoNet.Refine: A One-step Diffusion Model for Rapid Refinement of Structural Models with Cryo-EM Density Map Restraints},
  author={Huang, Fuyao and Yu, Xiaozhu and Xu, Kui and Zhang, Qiangfeng Cliff},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=NwzY2yhlme}
}
```

**Authors**: Fuyao Huang\*, Xiaozhu Yu\*, Kui Xu#, and Qiangfeng Cliff Zhang#  
**Paper**: [OpenReview](https://openreview.net/forum?id=NwzY2yhlme) | [PDF](https://openreview.net/pdf?id=NwzY2yhlme)

**Keywords**: Protein structure refinement; Cryo-electron microscopy; Deep learning; Density-guided refinement; Geometric restraints; Diffusion model

---



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

**Made with ❤️ by the CryoNet.Refine Team**

</div>
