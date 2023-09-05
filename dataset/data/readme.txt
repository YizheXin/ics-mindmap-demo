
## Overview

This repository contains datasets and training scripts for our machine learning project. The datasets are provided both from real and synthetic sources, and the script details the process of utilizing both types of datasets for model training.

## Directory Structure

### `/data`

This folder houses the real data sourced directly from our client. The primary dataset in this directory is:

- `real_data`: A subset of the actual data provided by our client, utilized for both training and testing purposes.

We have annotated this data using Label Studio and the annotated dataset is saved in the COCO format.

### `/synthetic_data`

This directory contains images that we have crafted. These images serve the purpose of generating an extensive set of synthetic data to supplement our real dataset.

### Training Notebook

- `13_train_on_synthetic_plus_real_data_v5.ipynb`: This Jupyter notebook serves two primary purposes:
  1. Generation of the synthetic dataset required for training.
  2. Model training on the combined dataset – a fusion of the synthetic dataset and the real data.
- 'trainer_yolos_real_data.ipynb'
  Use tensorflow to training own labeled real dataset(drawing,edges,etc..)

- Labeling more examples and retraining ViT models


This is a guideline for labeling more examples to get a performance gain in the three ViT models (drawing detection, edge tip detection and node detection)


There are train and test labeled examples for the three models in the repo in this folder:

https://github.com/addaxis/ics_mind-maps/tree/main/notebooks/data/real_data

There, you will find train and test labeled examples for the three models. 

To label more examples, Label Studio can be very useful. Start by following the installation instructions: https://labelstud.io/guide/get_started.html#Quick-start

The instructions are very straightforward. Import all new examples and select the task as computer vision->object detection with bounding boxes

Once all examples have been labeled, export the annotations in “COCO” format (Label Studio will give this as one of the possible options). Exporting the annotations in this format is very helpful as the training notebook is already prepared to take the examples in this format. Label Studio will create a folder and within it a Json file with the annotations and an image folder containing the examples. 

This folder can be used directly in the generic trainer notebook that can be found in the repository to train a model with the annotations: https://github.com/addaxis/ics_mind-maps/blob/main/notebooks/trainer_yolos_real_data.ipynb