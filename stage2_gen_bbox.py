# -*- coding: utf-8 -*-
#
# Image Level Label to Bounding Box (IL2BB) pipeline
# Author: Peter Ersts (ersts@amnh.org)
#
# --------------------------------------------------------------------------
#
# This file is part of Animal Detection Network's (Andenet)
# Image Level Label to Bounding Box (IL2BB) pipeline
#
# IL2BB is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IL2BB is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
import os
import sys
import csv
import json
import torch
import schema
import urllib.request
import numpy as np
from PIL import Image
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh

MEGADETECTOR = './md_v5a.0.0.pt'
MEGADETECTOR_URL = 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt'
THRESHOLD = 0.5

# Little error checking to get things started
if len(sys.argv) == 1:
    print('Usage: python3 {} batch_dir'.format(sys.argv[0]))
    sys.exit(0)
batch_path = sys.argv[1]
if not os.path.isdir(batch_path):
    print("[{}] is not valid directory.".format(batch_path))
    sys.exit(0)
csvfile = os.path.join(batch_path, "labels.csv")
if not os.path.isfile(csvfile):
    print("[{}] does not contain a label file.".format(batch_path))
    sys.exit(0)
logfile = os.path.join(batch_path, "log.txt")
batch_dir = os.path.split(batch_path)[1]
if batch_dir == '':
    batch_dir = os.path.split(os.path.split(batch_path)[0])[1]
bbxfile = os.path.join(batch_path, "{}_il2bb.bbx".format(batch_dir))

# Download frozen megadetector graph
if not os.path.isfile(MEGADETECTOR):
    print("Downloading Megadetector...")
    urllib.request.urlretrieve(MEGADETECTOR_URL, MEGADETECTOR)
    print("Download complete.")


# Open files
try:
    csvfile = open(csvfile, 'r')
    logfile = open(logfile, 'w')
    bbxfile = open(bbxfile, 'w')
except:
    print('Unable to open or create needed files')
    sys.exit(0)

# Grab list of images to process
nl = ''
image_list = []
reader = csv.reader(csvfile)
for row in reader:
    image_list.append((row[0], row[1]))
csvfile.close()
detections = schema.annotation_file()
detections['analysts'].append('Machine Generated')

# Check if GPUs are available
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'

try:
    if torch.backends.mps.is_built and torch.backends.mps.is_available():
        device = 'mps'
except AttributeError:
    pass

# Load model
checkpoint = torch.load('./md_v5a.0.0.pt')
# Patch for older YOLOV5 model
for m in checkpoint['model'].modules():
    if type(m) is torch.nn.Upsample:
        m.recompute_scale_factor = None
model = checkpoint['model'].float().fuse().eval().to(device)

# Pass each image through megadetector
for image_name, label in image_list:
    file_name = os.path.join(batch_path, image_name)
    img_original = Image.open(file_name)
    img_original = np.asarray(img_original)

    # padded resize
    img = letterbox(img_original, new_shape=1280, stride=64, auto=True)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255
    img = torch.unsqueeze(img, 0).to(device)
    pred: list = model(img)[0]
    pred = non_max_suppression(prediction=pred.cpu(), conf_thres=0.2)
    gn = torch.tensor(img_original.shape)[[1, 0, 1, 0]]

    entry = schema.annotation_file_entry()
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_original.shape).round()
            for *box, conf, cls in reversed(det):
                annotation = schema.annotation()
                annotation['created_by'] = 'machine'
                bbox = (xyxy2xywh(torch.tensor(box).view(1, 4)) / gn).view(-1).tolist()
                x_center, y_center, width_of_box, height_of_box = bbox
                x_min = x_center - width_of_box / 2.0
                y_min = y_center - height_of_box / 2.0
                x_max = x_center + width_of_box / 2.0
                y_max = y_center + height_of_box / 2.0
                annotation['bbox']['xmin'] = x_min
                annotation['bbox']['xmax'] = x_max
                annotation['bbox']['ymin'] = y_min
                annotation['bbox']['ymax'] = y_max
                annotation['label'] = label
                annotation['confidence'] = conf.item()
                entry['annotations'].append(annotation)
    # If bounding boxes created apply label else log false negative
    if len(entry['annotations']) > 0:
        detections['images'][image_name] = entry
        print("({}) [{}] detections".format(file_name, len(entry['annotations'])))
    else:
        print('False negative: [{}] - {}'.format(label, file_name))
        logfile.write('{}False negative: [{}] - {}'.format(nl, label, image_name))
        nl = "\n"

# Dump annotations and close open files
json.dump(detections, bbxfile)
bbxfile.close()
logfile.close()
