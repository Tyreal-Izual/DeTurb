# Atmospheric Turbulence Mitigation with Deformable 3D Convolutions and 3D Swin Transformers

## Overview

Atmospheric turbulence in long-range imaging significantly degrades the quality and fidelity of captured scenes due to random variations in both spatial and temporal dimensions. These distortions present a formidable challenge across various applications, from surveillance to astronomy, necessitating robust mitigation strategies. While model-based approaches achieve good results, they are very slow. Deep learning approaches show promise in image and video restoration but have struggled to address these spatiotemporal variant distortions effectively. This paper proposes a new framework that combines geometric restoration with an enhancement module. Random perturbations and geometric distortion are removed using a pyramid architecture with deformable 3D convolutions, resulting in aligned frames. These frames are then used to reconstruct a sharp, clear image via a multi-scale architecture of 3D Swin Transformers. The proposed framework demonstrates superior performance over the state of the art for both synthetic and real atmospheric turbulence effects, with reasonable speed and model size.

## Installation

Python 3.10+ with PyTorch. NVIDIA CUDA and CPU are supported; no custom CUDA extension is required.

```bash
conda create -n deturb python=3.11 -y
conda activate deturb
python -m pip install -e ".[metrics]"
```

## Models

| Model | Configuration | Weights |
| --- | --- | --- |
| Dynamic | `configs/dynamic.json` | [dynamic.pth](https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/dynamic.pth) |
| Static | `configs/static.json` | [static.pth](https://github.com/Tyreal-Izual/DeTurb/releases/download/v0.1.0/static.pth) |

Download the selected checkpoint into `weights/`. See [weight downloads](weights/README.md).

## Inference

```bash
deturb-infer --config configs/dynamic.json \
  --checkpoint weights/dynamic.pth \
  --input input.mp4 --output outputs/restored.mp4 --device cuda
```

For Static scenes, use `configs/static.json` and `weights/static.pth`. Output videos use MP4 without audio; odd dimensions are padded for encoding.

## Training

Set your local data paths:

```bash
export DYNAMIC_DATA_ROOT=/path/to/Deturb_dataset/dynamic
export STATIC_DATA_ROOT=/path/to/Deturb_dataset/static
```

### Single GPU

```bash
# Dynamic
python -m deturb.cli.train \
  --config configs/dynamic.json --manifest-path /path/to/dynamic.json \
  --batch-size 4

# Static
python -m deturb.cli.train \
  --config configs/static.json --manifest-path /path/to/static.json \
  --batch-size 4 --load weights/dynamic.pth --finetune
```

### Multiple GPUs

```bash
# Dynamic: 4 GPUs
torchrun --standalone --nproc_per_node=4 -m deturb.cli.train \
  --config configs/dynamic.json --manifest-path /path/to/dynamic.json \
  --batch-size 1

# Static: 4 GPUs
torchrun --standalone --nproc_per_node=4 -m deturb.cli.train \
  --config configs/static.json --manifest-path /path/to/static.json \
  --batch-size 1 --load weights/dynamic.pth --finetune
```

`--batch-size` is per GPU. Both examples use global batch 4: one GPU × 4 or four GPUs × 1. Single-GPU training requires more memory per GPU; reducing the batch size changes the global batch.

Use `--load /path/to/latest.pth` without `--finetune` to resume, keeping the same GPU count and per-GPU batch size. Config paths are relative to the JSON file; command-line paths are relative to the current directory. Outputs are saved under `outputs/`.

## Evaluation

```bash
deturb-metrics --config configs/dynamic.json \
  --checkpoint weights/dynamic.pth \
  --manifest /path/to/evaluation.json \
  --output outputs/metrics.json --device cuda
```

Reports PSNR, SSIM, AlexNet LPIPS/tLPIPS and GT-flow Farneback warping error before video encoding. The first four metrics use equal video weights; warping error weights valid pixels.

See [metric dependencies](requirements-metrics.txt).

## Citation and acknowledgements

```bibtex
@inproceedings{zou2024deturb,
  title={DeTurb: atmospheric turbulence mitigation with deformable 3D convolutions and 3D Swin transformers},
  author={Zou, Zhicheng and Anantrasirichai, Nantheera},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={904--921},
  year={2024}
}
```

See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
