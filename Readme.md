# Image Level Label to Bounding Box Pipeline
The Image Level Label to Bounding Box (IL2BB) pipeline automates the generation of labeled bounding boxes by leveraging an organization’s previous labeling efforts and [Microsoft AI for Earth’s MegaDetector](https://github.com/microsoft/CameraTraps/). The output of this pipeline are batches of images with annotation files that can be opened, reviewed, and modified with the [Bounding Box Editor and Exporter (BBoxEE)](https://github.com/persts/BBoxEE) to prepare training data for object detectors.

The IL2BB pipeline is especially useful for organizations that are hesitant or not permitted to use or store data on online services.

## Problem Statment
Most organizations undertaking camera trap initiatives don’t have the human capital to collect and label bounding boxes needed to train deep learning based object detectors let alone add bounding boxes to historical / previously labeled images.

## Context
Camera traps are one of the most valuable tools used by wildlife biologists, managers, and conservation practitioners for wildlife research and monitoring. Any analysis of camera trap data involves two core activities; 1) reviewing each image captured and 2) documenting what, if anything, appears in each image. 

Regardless of what data are recorded and how they are stored (flat file, Excel spreadsheet, database etc.), two fields exist in every camera trap dataset; 1) image file name and 2) the name, which from here on will be called the label, of the object(s) captured in the image.

These image file name and label pairs are precisely what are needed as input to train powerful neural network based image classifiers. Object detectors provide even more useful information (e.g., location and counts of targets) but require additional training data. Specifically, deep learning based object detectors require a bounding box and label for each desired target present in an image.

Reviewing and documenting objects captured in images is time consuming enough. Collecting and labeling bounding boxes requires even more time.

## Relevance
Deep learning based object detectors have the potential to assist and automate the analysis of images collected during camera trap deployments. Considerable bounding box data are needed to train an object detector. Most organizations, however, don’t have the human capital to manually generate the needed training data let alone reprocess historical or previously labeled images. An automated pipeline to convert existing image level labels into labeled bounding boxes would give organizations a tremendous boost toward training custom networks to assist with the analysis of newly collected data.

# Getting Started
The Il2BB pipeline was developed with Python 3.10.12 on Ubuntu 22.04.

## Set up a virtual environment
```bash
cd [IL2BB Workspace]

[Linux]
git clone https://github.com/persts/IL2BB IL2BB
python3 -m venv il2bb_env
source il2bb_env/bin/activate

[Windows]
git clone https://github.com/persts/IL2BB IL2BB
python -m venv il2bb_env
il2bb_env\Scripts\activate


cd IL2BB
python -m pip install pip --upgrade
python -m pip install -r requirements.txt
```

## Quick Start
The [Colorado Parks and Wildlife use case](./UseCase) doubles as basic user guide.

