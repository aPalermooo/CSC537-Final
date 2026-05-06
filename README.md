# License Plate Segmentation and Classification

**Author:** Xander Palermo  
**Class:** CSC537 - Deep Learning  
**Instructor:** Mukulika Ghosh
**Last Accessed:** 6 May 2026

This project focuses on completing a full pipeline that processes an image of a car, and identifies and classifies
characters on its license plate into their ASCII representation. This technology has wide applications in traffic regulation,
namely administering toll fees, administering speeding tickets, and surveillance.

----

## Approach

This problem can be broken down into 3 phases:
1. License Plate Segmentation
2. Character Segmentation
3. Character Classification

To tackle each problem, a model was designed and iterated using different architectures for each phase. They are separated
into their own respective directories and designed to be interchangeable in the final pipeline implementation.
---

## Replication
The following steps allow for the replication of the training of models created

### Project Structure

To maintain the goal of creating interchangeable parts, the use of the project structure was developed to aid in keeping track of each
component needed. Each directory contains a __init__.py file to configure the directory into a python package, to simplify the logic
when calling to run a specific file. The main components are:  
  
**Compilers:** Transform the raw dataset into dataloader objects useful by PyTorch. Will also restructure the raw dataset into a more
workable form in their main functions, i.e. splitting the data into training and testing samples, changing annotation formats.  
This folder also has a helper directory used to house shared logic that is used across compilers, and custom class implementations of datasets.
  
**Datasets:** Contains the dataset used to train each model used within the pipeline (if applicable). Data 
from Kaggle can be dumped into raw directory and compiler will handle from there.  
  
**Models:** Contains the implementation of all the models used within the pipeline. This manifests as describing 
the architecture and specific necessary overrides to become compatible with the rest of the pipeline.
  
**Pipelines:** Contain complete implementations of transforming a random image from plate_segmentation dataset into the target
ASCII representation of its license plate.

**Results:** The output folder of model training. Used to store weights and statistics of performance of different models for use in a given Pipeline.

**Training:** Take the dataloaders from the compilers and complete forward and backward passes through models to train and test them.

*The required python packages are described in requirements.txt*

Most of these directories are divided into the 3 phases, and further contained python files labeled with their iteration number.
Each File has a note commented at the top of the file that describes the changes made from the previous iteration.

### Datasets

The following datasets were utilized to train and test the solutions designed for this project. To import them into the 
project, each one was downloaded as a zip file and their contents were dumped into their respective dataset/raw directory.

> **License Plate Segmentation/Complete Pipeline:**
> [Dataset Link](https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition)  
> By: Adil Shamim
> 10,000 Images of cars (with license plates) from various regions around the world.  
> Have variation in lighting conditions and camera qualities to provide better generalization
> 
> **Character Segmentation:**
> [Dataset Link](https://www.kaggle.com/datasets/francescopettini/license-plate-characters-detection-ocr)  
> By: Francesco Pettini
> 209 images containing 2026 bounding boxes and labeled characters
> 
> **Character Classification:**
> [Dataset Link](https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset)  
> By: Jelal  
> 35,000 images of characters found on license place labeled by ASCII representation (1,000 images per class)


### Execution

For assembling a pipeline the following stages need to be complete  
Running the target compiler for each phase → Running the training loop for each prescribed model → Running the final pipeline file.  
  
To run a specific file call it from the project root directory as a module ex: 
``python3 -m training.character_classification.training_loop_v1``  
substituting the package name out for the intended file you need to run
