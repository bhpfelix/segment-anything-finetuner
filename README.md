# Simple Finetuner for Segment Anything

This repository contains a simple starter code for finetuning the [FAIR Segment Anything](https://github.com/facebookresearch/segment-anything) (SAM) models leveraging the convenience of [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

### Setup

1. Install dependencies

    First run
    
    ```bash
    git clone --recurse-submodules git@github.com:bhpfelix/segment-anything-finetuner.git
    ```
    
    Then
    
    ```bash
    cd segment-anything-finetuner
    ```

    Follow the [setup instruction](https://github.com/facebookresearch/segment-anything/blob/main/README.md#installation) of Segment Anything to install the proper dependencies. Then run

    ```bash
    pip install -r requirements.txt
    ```

2. Data preparation

    The starter code supports Coco format input with the following layout

    ```yaml
    ├── dataset_name/
    │   ├── train/
    │   │   ├── _annotations.coco.json # COCO format annotation
    │   │   ├── 000001.png             # Images
    │   │   ├── 000002.png
    │   │   ├── ...
    │   ├── val/
    │   │   ├── _annotations.coco.json # COCO format annotation
    │   │   ├── xxxxxx.png             # Images
    │   │   ├── ...
    ```

3. Download model checkpoints

    Download the necessary SAM model checkpoints and arrange the repo as follows:

    ```yaml
    ├── dataset_name/              # structure as detailed above
    │   ├── ...
    ├── segment-anything/          # The FAIR SAM repo
    │   ├── ...
    ├── SAM/                       # the SAM pretrained checkpoints
    │   ├── sam_vit_h_4b8939.pth
    │   ├── ...
    ├── finetune.py
    ├── ...
    ```

### Finetuning (`finetune.py`)

This file contains a simple finetuning script for the Segment Anything model on Coco format datasets.

Example usage:

```python
python finetune.py \
    --data_root ./dataset_name \
    --model_type vit_h \
    --checkpoint_path ./SAM/sam_vit_h_4b8939.pth \
    --freeze_image_encoder \
    --batch_size 2 \
    --image_size 1024 \
    --steps 1500 \
    --learning_rate 1.e-5 \
    --weight_decay 0.01
```

We can optionally use the `--freeze_image_encoder` flag to detach the image encoder parameters from optimization and save GPU memory.

### Notes
- As of now the image resizing implementation is different from the `ResizeLongestSide` transform in SAM.
- Drop path and layer-wise learning rate decay are not currently applied.
- The finetuning script currently only supports bounding box input prompts.

### Resources
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch)
- [Detectron2](https://github.com/facebookresearch/detectron2)

### Citation

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
