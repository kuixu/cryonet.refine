## CryoNet.Refine

**CryoNet.Refine**, a new AI-driven real-space refinement framework for proteins, DNA, RNA, and their complexes. CryoNet.Refine employs a **one-step diffusion model** that tightly integrates experimental density information with stereochemical and physical constraints, enabling rapid convergence to structures that simultaneously exhibit high model–map correlation and model geometric metrics. Across comprehensive benchmarks, CryoNet.Refine delivers **substantial improvements** over traditional refinement methods in both model–map correlation and model geometric metrics, while achieving significantly lower computational cost. By providing an automated, scalable, and robust refinement pipeline, CryoNet.Refine represents an important step toward next-generation cryo-EM model refinement and broadens the applicability of AI-assisted structural biology.

---

## 1. Installation

### 1.1 Create Conda environment

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

For users in China, you can alternatively use the Aliyun mirror (example):

```bash
# pip install torch==2.5.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu12
```

### 1.3 Install cctbx chem_data

CryoNet.Refine uses cctbx geometry libraries and requires `chem_data`:

```bash
wget https://github.com/cctbx/cctbx_project/releases/download/v2025.10/chem_data-2025.10-pyhe0d8492_0.conda
conda install ./chem_data-2025.10-pyhe0d8492_0.conda
```

Make sure this is done inside the `cryonet.refine` conda environment.

---

## 2. Example Usage

From the project root (`cryonet.refine`), run:

```bash
sh ./run.sh ./examples/6ksw_af3.cif ./examples/6ksw.mrc 3.6 /output
```

Arguments:

- `./examples/6ksw_af3.cif` – Input model (PDB/mmCIF) to be refined
- `./examples/6ksw.mrc` – Input cryo-EM density map
- `3.6` – Map resolution (Å)
- `/output` – Output directory where refined models and logs will be written

The script will run the full refinement pipeline and save refined structures and metrics into `/output`.

---

## 3. Project Structure

- `CryoNetRefine/` – Core library
  - `libs/` – Geometry, protein representation, density utilities
  - `model/` – Network architecture and refinement engine
  - `data/` – Feature generation, IO, and preprocessing
- `examples/` – Example input structure and map
- `run.sh` – Convenience script for running refinement
- `cryonet.refine_env.yml` – Conda environment specification

---

## 4. Acknowledgements

CryoNet.Refine borrows code from the **Boltz** project:

Repository: `https://github.com/jwohlwend/boltz.git`

We have adapted and extended parts of their codebase and finetuned the model for our specific refinement setting. We gratefully acknowledge the Boltz authors for making their work available.

