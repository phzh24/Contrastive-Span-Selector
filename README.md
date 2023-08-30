# Contrastive-Span-Selector

This repo provides the source code of our PRICAI-2023 paper:
*CSS: Contrastive Span Selector for Multi-span Question Answering*.

This study investigates the [Multi-span Question Answering](https://multi-span.github.io/) task and achieves state-of-the-art performance.


## Environment

```python
datasets==2.12.0
numpy==1.24.3
python==3.10.0
torch==2.0.1+cu118
torch-scatter==2.1.1+pt20cu118
transformers==4.28.1
```
> Note: To install `torch-scatter`, please refer to the installation instructions on the [torch-scatter · PyPI](https://pypi.org/project/torch-scatter/) page.

## Data Preprocessing

1. Download MultiSpanQA dataset from [MultiSpanQA Homepage](https://multi-span.github.io/).
2. Move the two dataset settings to `./data/{DATASET_SETTING}/raw`, respectively.
3. Run 
```sh
python ./data/preprocess.py --dataset_setting "{DATASET_SETTING}"
```

> DATASET_SETTING: multispan, or expanded.


## Training

>Note: Please update the code if you're not using `wandb` to log experiment data.

Multiple GPUs:

1. Run
```sh
export WANDB_PROJECT=Contrastive-Span-Selector
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
torchrun --nproc_per_node 2 ./src/main.py ./config/multispan-bert.json
```

Single GPU:

1. In file `./config/multispan-bert.json`, change the `gradient_accumulation_steps` parameter.
2. Run
```sh
export WANDB_PROJECT=Contrastive-Span-Selector
export CUDA_VISIBLE_DEVICES=0
python ./src/main.py ./config/multispan-bert.json
```

## Prediction

1. In file `./config/multispan-bert-prediction.json`, change the `output_dir` parameter to the directory of the best-performing checkpoint.
2. Run
```sh
export CUDA_VISIBLE_DEVICES=0
python ./src/main.py ./config/multispan-bert-prediction.json
```
