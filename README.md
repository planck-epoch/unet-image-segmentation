# IDCARD_SEGMENTATION

**Semantig Segmentation for ID Card**

This repository contains algortihm to build semantig segmentation for ID Card using U-Net including labeller application to anotate the bounding box of id cards.

## Datasets

Put your datasets in the following directory structure
```
.
+-- dataset
|   +-- data
|       +-- ina_id
|           +-- ground_truth
|           +-- images
+-- labeler
+-- utils
```
- Download dataset :

    [MIDV-500](https://arxiv.org/abs/1807.05786)

- Create your own dataset :
    
    ```
    jupyter notebook "labeler/labeler-interface.ipynb"
    ```

## Installation
1. Create and activate a new environment.
```
conda create -n idcard python=3.6
source activate idcard
```
2. Install Dependencies.
```
pip install -r requirements.txt
```
## Prepare Dataset
Splits the data into training, test and validation data.
```
python prepare_dataset.py
```

### Training of the neural network
```
python train.py
```

### Show Jupyter Notebook for Test
```
jupyter notebook "IDCard Prediction Test.ipynb"
```

### Evaluate model
```
python benchmark.py ./dataset/data/ina_id_selfie --model ./model/model_selfie_3.h5 --threshold 0.9
```