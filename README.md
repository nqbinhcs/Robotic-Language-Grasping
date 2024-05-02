# Language-Driven Grasping
This is the repository of work "TransGRNet: Generative Residual Convolutional Neural Network with
Transformer for Language-Driven Grasping"
## Table of contents
   1. [Installation](#installation)
   1. [Datasets](#datasets)
   1. [Training](#training)
   1. [Testing](#testing)

## Installation
- Create a virtual environment
```bash
$ conda create -n transgrnet python=3.9
$ conda activate transgrnet
```

- Install pytorch
```bash
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```

## Datasets
- The dataset should be include in the following hierarchy:
```
train_data/grasp-anything
|-- seen
|   |-- grasp_instructions
|   |-- grasp_label
|   `-- image
`-- unseen
    |-- grasp_instructions
    |-- grasp_label
    `-- image
```

## Training
To reproduce our results with TransGRNet, use the following training command:
```bash
$ python train_network.py --dataset grasp-anything --dataset-path train_data/grasp-anything/seen --network trans_grconvnet --use-instruction --use-depth 0 --batch-size 16 --vis --lr 0.0001 --dropout-prob 0.3 --pretrained
```
Additionally, the weight of this training process is provided in `weights/best_trans_grconvnet`


## Testing
For the testing procedure, execute the following command:
```bash
$ python evaluate.py --network weights/best_trans_grconvnet --dataset grasp-anything --dataset-path train_data/grasp-anything/unseen --iou-eval --use-depth 0  --n-grasps 1 --use-instruction 
```


## Acknowledgement
Our codebase is developed based on [Vuong et al.](https://github.com/andvg3/Grasp-Anything).
