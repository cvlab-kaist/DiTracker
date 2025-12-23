# Repurposing Video Diffusion Transformers for Robust Point Tracking

<p align="center">
  <a href="https://scholar.google.com/citations?hl=&user=Eo87mRsAAAAJ"><strong>Soowon Son</strong></a><sup>1</sup> ·
  <a href="https://hg010303.github.io/"><strong>Honggyu An</strong></a><sup>1</sup> ·
  <a href="https://kchyun.github.io/"><strong>Chaehyun Kim</strong></a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=oh5Od2wAAAAJ"><strong>Hyunah Ko</strong></a><sup>1</sup> ·
  <a href="https://nam-jisu.github.io/"><strong>Jisu Nam</strong></a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?hl=&user=EU52riMAAAAJ"><strong>Dahyun Chung</strong></a><sup>1</sup> · <br>
  <a href="https://scholar.google.com/citations?hl=&user=rXRHxkwAAAAJ"><strong>Siyoon Jin</strong></a><sup>1</sup> ·
  <a href="https://yj-142150.github.io/jungyi/"><strong>Jung Yi</strong></a><sup>1</sup> ·
  <a href="https://scholar.google.com/citations?user=WIiNrmoAAAAJ"><strong>Jaewon Min</strong></a><sup>1</sup> ·
  <a href="https://hurjunhwa.github.io/"><strong>Junhwa Hur</strong></a><sup>2†</sup> ·
  <a href="https://cvlab.kaist.ac.kr"><strong>Seungryong Kim</strong></a><sup>1†</sup>
</p>

<p align="center">
  <sup>1</sup>KAIST AI &nbsp;&nbsp;&nbsp;&nbsp; <sup>2</sup>Google DeepMind
</p>

<p align="center">
  <sup>†</sup>Co-corresponding authors
</p>

<p align="center">
  <a href="https://cvlab-kaist.github.io/DiTracker/">
    <img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/XXXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
</p>

<p align="center">
  <img src="assets/main_architecture.png" width="100%">
</p>

> TL;DR: **DiTracker** repurposes video Diffusion Transformers (DiTs) for point tracking with softmax-based matching, LoRA adaptation, and cost fusion, achieving **stronger robustness and faster convergence** on challenging benchmarks.

<br>

## 🔧 Environment Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/cvlab-kaist/DiTracker.git
cd DiTracker

conda create -n DiTracker python=3.11 -y
conda activate DiTracker
pip install -r requirements.txt
pip install -e .

# Install modified diffusers library
cd diffusers
pip install -e .
cd ..
```


## 📁 Dataset Preparation

### Evaluation Datasets
Download the following datasets for evaluation:
- [**TAP-Vid-DAVIS & TAP-Vid-Kinetics**](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid)
- [**ITTO-MOSE**](https://huggingface.co/datasets/demalenk/itto-dataset)

Organize the datasets with the following directory structure:

```
/path/to/data/
├── tapvid/
│   ├── davis/
│   └── kinetics/
└── itto/
    └── mose/
```

### Training Dataset

For training, we use the **Kubric-MOVi-F** dataset from CoTracker3. Download [CoTracker3 Kubric Dataset](https://huggingface.co/datasets/facebook/CoTracker3_Kubric)



## 🚀 Inference

Pre-trained DiTracker weights are included in the `./checkpoint` directory. Use these weights to evaluate on various benchmarks and challenging scenarios.

### Evaluation on Benchmarks

Run the following commands to evaluate DiTracker on different benchmarks:

```bash
# ITTO-MOSE
python evaluate.py --config-name eval_itto_mose_first dataset_root=/path/to/data

# TAP-Vid-DAVIS
python evaluate.py --config-name eval_tapvid_davis_first dataset_root=/path/to/data

# TAP-Vid-Kinetics
python evaluate.py --config-name eval_tapvid_kinetics_first dataset_root=/path/to/data
```

**Note:** ITTO-MOSE evaluation includes detailed metrics on motion dynamics and reappearance frequency.

### Evaluation on Corruptions

Test robustness under various ImageNet-C corruption types:

```bash
python evaluate.py dataset_root=/path/to/data severity=5
```
- `severity`: Corruption intensity. Higher values indicate stronger corruption.

### Visualization

To visualize tracked trajectories, add the `visualize=True` option:

```bash
python evaluate.py --config-name eval_itto_mose_first dataset_root=/path/to/data visualize=True
```


## 🏋️ Training

To train DiTracker from scratch:

```bash
python train.py --ckpt_path ./output --dataset_root /path/to/data
```

All training parameters are configured to match the paper's specifications. Experiments were conducted on **NVIDIA RTX A6000** GPUs.

### Key Training Parameters

Other parameters can be customized. But for best performance, we recommend keeping these parameters at their default values as described in the paper.

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `--model_path` | `CogVideoX-2B` | Video DiT backbone model |
| `--layer_hooks` | `[17]` | Layer indices in video DiT for query-key extraction |
| `--head_hooks` | `[2]` | Attention head indices for query-key extraction |
| `--model_resolution` | `[480, 720]` | Input resolution (height × width) |
| `--cost_softmax` | `True` | Use softmax for cost calculation (vs. normalized dot product) |
| `--resnet_fuse_mode` | `"concat"` | ResNet fusion: `"add"` (average), `"concat"`, or `None` (disable) |


## 🙏 Acknowledgements

This code is built upon [CoTracker3](https://github.com/facebookresearch/co-tracker). We sincerely thank the authors for their excellent work and for making their code publicly available.



## 📝 Citation

If you find DiTracker useful for your research, please consider citing:

```bibtex
@article{son2025ditracker,
  title={DiTracker: Repurposing Video Diffusion Transformers for Robust Point Tracking},
  author={Son, Soowon and An, Honggyu and Kim, Chaehyun and Ko, Hyunah and Nam, Jisu and Chung, Dahyun and Jin, Siyoon and Yi, Jung and Min, Jaewon and Hur, Junhwa and Kim, Seungryong},
  journal={arXiv preprint arXiv:XXXXX.XXXXX},
  year={2025}
}
```