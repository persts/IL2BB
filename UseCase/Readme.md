# USE CASE: Colorado Parks and Wildlife

## Description
Colorado Parks and Wildlife (CPW) participated in an early effort to develop a machine learning approach that could be applied across study sites and provide a workflow that would allow ecologists to automatically identify wildlife in their own camera tap images ( [Machine learning to classify animal species in camera trap images: Applications in ecology](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13120) ). CPW contributed 329,688 images with image level labels to the development of the neural network based image classifier and these data are now part of the publicly available [North American Camera Trap Image](http://lila.science/datasets/nacti) archive. 

CPW would now like to develop a more advanced deep learning based object detector to help process images collected during future camera trap deployments. While CPW has an extensive camera trap archive, CPW does not have the resources to re export the data and manually collect and label the needed bounding boxes.

CPW will use the [IL2BB](https://github.com/persts/IL2BB) pipeline to leverage their previous labeling and data collation efforts (available the in NACTI archive) and automatically create labeled bounding boxes to train a new deep learning based object detector.

## Step 1: Downloading the NACTI Data & Metadata
All of the CPW images are stored in the first archive chunk.

The needed files include:
* https://lilablobssc.blob.core.windows.net/nacti/nactiPart0.zip
* https://lilablobssc.blob.core.windows.net/nacti/nacti_metadata.csv.zip

nactiPar0.zip is 488GB! Even through it appears like you can access the link through your browser, to successfully download the data you must use the [Azure (azcopy)](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) tool.

*Note: The [UseCase] directory contains five metadata entries and images from the NACTI for demonstration purposes. You do not need to download the original data to test the pipeline! Just
```code
cd UseCase
```
to get started.

## Step 2: Extracting CPW Metadata
The North American Camera Trap Image (NACTI) archive contains 3,367,383 images in total. The metadata for this archive contains fourteen fields but the IL2BB pipeline only requires two fields (image file name and object label). 

```python
python3 extract_metadata.py
```

## Step 3: Creating Image Batches
This is the step most organizations will start with if they are exporting data from their own internal database(s) or archives.

### Label Map
The main input to the batching stage is a simple csv file (no header row) with two fields:
 1. image file name
 2. label
 
Example:
```code
part0/sub000/2010_Unit150_Ivan097_img0001.jpg,red deer
part0/sub000/2010_Unit150_Ivan097_img0002.jpg,red deer
part0/sub000/2010_Unit150_Ivan097_img0003.jpg,red deer
part0/sub000/2010_Unit150_Ivan099_img0121.jpg,red deer
part0/sub000/2010_Unit150_Ivan099_img0122.jpg,red deer
```

*Note: The IL2BB pipeline expects all of the targets in the image to be of the same species. When creating your label map, skip any images that have more than one species in them. Images can have multiple targets of the same species.  


Usage:
```python
python3 stage1_batch.py [label_map] [data_dir] [batch_dir]
```

The batching process will copy the images from their original location (i.e., column 1 of the label map file) into a series of sequentially number directories with 1000 images per directory. The original files are renamed during the copy process and a local label map file is created in each batch directory.

The new label map file has three fields:
 1. new image file name
 2. label
 3. original image file name

Run the batch process on the demo data.
```python
python3 ../stage1_batch.py cpw_labelmap.csv data batches
```
## Step 4: Creating Bounding Boxes
The last step is to use [Microsoft AI for Earth’s MegaDetector](https://github.com/microsoft/CameraTraps/) for creating an initial bounding box and then apply the appropriate label to the box based on the known image level label. 

```python
python3 ../stage2_gen_bbox.py batches/batch_001/
```
This step will create a new annotation file (il2bb.bbx) in the batch directory which can be opened, reviewed, and modified with the [Bounding Box Editor and Exporter (BBoxEE)](https://github.com/persts/BBoxEE). A log file is also generated in the batch directory which contains entries for images where no bounding box was able to be created.

Once all of the data have been reviewed, BBoxEE can be used to generate training data for several popular object detector training pipelines.