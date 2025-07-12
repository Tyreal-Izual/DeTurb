# Atmospheric Turbulence Mitigation with Deformable 3D Convolutions and 3D Swin Transformers

## Overview
Atmospheric turbulence in long-range imaging significantly degrades the quality and fidelity of captured scenes due to random variations in both spatial and temporal dimensions. These distortions present a formidable challenge across various applications, from surveillance to astronomy, necessitating robust mitigation strategies. While model-based approaches achieve good results, they are very slow. Deep learning approaches show promise in image and video restoration but have struggled to address these spatiotemporal variant distortions effectively. This paper proposes a new framework that combines geometric restoration with an enhancement module. Random perturbations and geometric distortion are removed using a pyramid architecture with deformable 3D convolutions, resulting in aligned frames. These frames are then used to reconstruct a sharp, clear image via a multi-scale architecture of 3D Swin Transformers. The proposed framework demonstrates superior performance over the state of the art for both synthetic and real atmospheric turbulence effects, with reasonable speed and model size.


## Installation

This project is developed under an Anaconda environment, leveraging Python 3.8.18 and CUDA 12.1. To set up the project environment:

1. Navigate to the `code` directory:
    ```bash
    cd code
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training Instructions

### For Dynamic Scenes

#### Full Model Training (Recommended)

```bash
python train_TMT_2stage.py --batch-size 1 --patch-size 256 --train_path "path of training videos folder" --val_path "path of validation videos folder" --log_path "path to save logging files and images"
```

<details>
<summary> Train 1-Stage Model </summary>

```bash
python train_TMT_1stage.py --batch-size 1 --patch-size 256 --train_path "path of training videos folder" --val_path "path of validation videos folder" --log_path "path to save logging files and images"
```
</details>


### For Static Scenes

#### Full Model Training (Recommended)
```bash
python train_TMT_2static.py --batch-size 1 --patch-size 256 --train_path "path of training imgs folder" --val_path "path of validation imgs folder" --log_path "path to save logging files and images"
```

<details>
<summary> Train 1-Stage Model </summary>

```bash
python train_TMT_1static.py --batch-size 1 --patch-size 256 --train_path "path of training imgs folder" --val_path "path of validation imgs folder" --log_path "path to save logging files and images"
```

```bash
```
</details>


### Note
Please change the absolute path of `model` folder in file `code/model/TMT_DC.py` and `code/model/TMT_DC2.py` when you training. (should be line 11 and 13, shown as `sys.path.append('C:\\Users\\Zouzh\\Desktop\\IP\\code\\model')`)  

If encountering a CUDA out of memory error, reduce the patch size to no less than 128 to ensure model quality.


## Video Reconstruction (single video)

#### Full Stage Model
```bash
python video_inference2.py --input_path 'path of input video' --out_path 'path of output video' --model_path 'Load model from a .pth file' --save_video
```

<details>
<summary> 1-Stage Model </summary>

```bash
python video_inference.py --input_path 'path of input video' --out_path 'path of output video' --model_path 'Load model from a .pth file' --save_video
```
</details>


### Note
For the `out_path`, please remember to contain the output file name you want, for example: `'code/log/out.mp4'` 

## Video Reconstruction (videos in folder)

#### Full Stage Model
```bash
python video_inferencefolder2.py --input_path 'path of input video' --out_path 'path of output video' --model_path 'Load model from a .pth file' --save_video
```
<details>
<summary> 1-Stage Model </summary>

```bash
python video_inferencefolder.py --input_path 'path of input video' --out_path 'path of output video' --model_path 'Load model from a .pth file' --save_video
```
</details>


## Evaluation

There are some helpful evaluation tools under `code/log/tools`. 

## Citation
If you find our code implementation helpful for your own research or work, please cite our work: 

```
@inproceedings{zou2024deturb,
  title={DeTurb: atmospheric turbulence mitigation with deformable 3D convolutions and 3D Swin transformers},
  author={Zou, Zhicheng and Anantrasirichai, Nantheera},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={904--921},
  year={2024}
}
```

## Acknowledgements

This project is built upon code from [TMT](https://github.com/xg416/TMT). We have made modifications to the original code, and these changes are distributed under the Apache-2.0 License. 

## License

This project is based on [TMT](https://github.com/xg416/TMT), which does not specify a license. Our modifications to this project are licensed under the [Apache-2.0 License](https://github.com/Tyreal-Izual/Atmosphere-Turbulence-Mitigation/blob/main/LICENSE).

```bash
Copyright 2024 Frederick Zou
```

For more information, please see the [LICENSE](https://github.com/Tyreal-Izual/Atmosphere-Turbulence-Mitigation/blob/main/LICENSE) file.


