# Language Modeling of Antibody Repertoires Identifies Blood-Accessible Immune Signatures of Alzheimer’s Disease

This repository provides a modular pipeline for antibody sequence analysis using pretrained antibody language models.  
It is part of our research project on using deep learning for Alzheimer's disease (AD) classification from antibody repertoires.
## Related Paper

This code accompanies our research paper:

> **[Language Modeling of Antibody Repertoires Identifies Blood-Accessible Immune Signatures of Alzheimer’s Disease]**  
> Authors: Chiu-Wang Tseng, Ya-Hui Chang, Hui-Chung Kuan, Yu Sun, Cheng-Ying Chou, Hong-Hsing Liu  
> Journal [TBD], Year [TBD] 
> [Link to the paper](https://doi.org/XXXXXXX)

Please cite our work if you find this repository useful in your research:

```bibtex
@article{hhliulabnhri2025ablmad,
  title   = {Language Modeling of Antibody Repertoires Identifies Blood-Accessible Immune Signatures of Alzheimer’s Disease},
  author  = {Chiu-Wang Tseng and Ya-Hui Chang and Hui-Chung Kuan and Yu Sun and Cheng-Ying Chou and Hong-Hsing Liu},
  journal = {TBD},
  year    = {TBD},
  note    = {Code available at \url{https://github.com/HHLiuLab-NHRI/AbLM-AD-Classifier}}
}
```
---
## Installation 

### Prerequisites

To run this project successfully using the provided Docker environment, your machine must meet the following prerequisites:

#### 1. Hardware
* **GPU:** An NVIDIA GPU is required. Due to the demands of modern deep learning models, an **NVIDIA Ampere architecture (or newer)** is strongly recommended (e.g., A100, H100, RTX 30 Series, RTX 40 Series).

#### 2. Software
* **Operating System:** Ubuntu is highly recommended for best compatibility with Docker and GPU drivers.
* **NVIDIA Driver:** The latest stable NVIDIA proprietary drivers must be installed for your GPU.
* **Docker Engine:** Docker must be installed and running on your system.
* **NVIDIA Container Toolkit:** This toolkit is essential for enabling the Docker container to access and utilize your NVIDIA GPU(s).

### Environment Setup

> **Important:** It is **highly recommended** to run all steps inside a Docker container to ensure a consistent and reproducible environment that includes the necessary versions of PyTorch, Transformers, and CUDA.

We will use the official **Hugging Face PyTorch GPU image** as our base: `docker.io/huggingface/transformers-pytorch-gpu:4.29.1`.

#### 1. Clone the Repository

First, clone this repository to your local machine. You must choose a path (e.g., `/home/user/projects/AbLM-AD-Classifier`) and replace `/YOUR/MOUNTED/HOST/PATH` with this actual path in the following steps.

```bash
git clone https://github.com/HHLiuLab-NHRI/AbLM-AD-Classifier /YOUR/MOUNTED/HOST/PATH
```

#### 2. Start the Docker Environment
Use the command below to start the container. This command utilizes the NVIDIA Container Toolkit (--gpus all) and sets up a Jupyter Lab environment for easy development.

Be sure to replace /YOUR/MOUNTED/HOST/PATH with the same path used in step 1.

```bash
docker run -it --rm --gpus all --shm-size 2g -p 8888:8888 \
-v /YOUR/MOUNTED/HOST/PATH:/workspace/host/ \
-w /workspace/host \
docker.io/huggingface/transformers-pytorch-gpu:4.29.1 \
bash -c "pip install jupyterlab && jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=''"
```

#### 3. Access and Start the Project
**Access Jupyter Lab**: Once the container is running, open your web browser and navigate to: http://localhost:8888

 The Jupyter Lab interface will open directly to your cloned directory, which is set as the working directory (/workspace/host). Locate and open the main tutorial notebook to begin:

- `tutorial_usage.ipynb`

---
## Pretrained Antibody Language Model

This repository **requires** a pre-trained antibody language model.

We recommend referring to [**AntiBERTa**](https://github.com/alchemab/antiberta) for a suitable model and instructions on its training or acquisition.

---
## License

The default `LICENSE` file in this repository is a placeholder for **academic / non‑commercial use** only.  

You can refine or replace it later (for example to a more formal institutional academic license, or another license of your choice).





