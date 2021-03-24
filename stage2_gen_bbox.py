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
import urllib.request
import numpy as np
from PIL import Image
import schema

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf

MEGADETECTOR = './md_v4.1.0.pb'
MEGADETECTOR_URL = 'https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb'
THRESHOLD = 0.8

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

# Create the detection graph and read in megadetector
detection_graph = tf.Graph()
graph_def = tf.GraphDef()
with tf.io.gfile.GFile(MEGADETECTOR, 'rb') as fid:
    serialized_graph = fid.read()
    graph_def.ParseFromString(serialized_graph)

with detection_graph.as_default():
    # Import graph
    tf.import_graph_def(graph_def, name='')

    # Begin processing loop
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = (detection_graph.get_tensor_by_name('image_tensor:0'))
        d_boxes = (detection_graph.get_tensor_by_name('detection_boxes:0'))
        d_scores = (detection_graph.get_tensor_by_name('detection_scores:0'))
        d_classes = (detection_graph.get_tensor_by_name('detection_classes:0'))
        num_detections = (detection_graph.get_tensor_by_name('num_detections:0'))

        # Pass each image through megadetector
        for img, label in image_list:
            file_name = os.path.join(batch_path, img)
            image = Image.open(file_name)
            image_np = np.array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image.close()
            fd = {image_tensor: image_np_expanded}
            (boxes, scores, classes, num) = sess.run([d_boxes,
                                                      d_scores,
                                                      d_classes,
                                                      num_detections],
                                                      feed_dict=fd)
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            entry = schema.annotation_file_entry()
            for i in range(len(scores)):
                if scores[i] >= THRESHOLD:
                    annotation = schema.annotation()
                    annotation['created_by'] = 'machine'
                    bbox = boxes[i]
                    annotation['bbox']['xmin'] = float(bbox[1])
                    annotation['bbox']['xmax'] = float(bbox[3])
                    annotation['bbox']['ymin'] = float(bbox[0])
                    annotation['bbox']['ymax'] = float(bbox[2])
                    annotation['label'] = label
                    entry['annotations'].append(annotation)
            # If bounding boxes created apply label else log false negative
            if len(entry['annotations']) > 0:
                detections['images'][img] = entry
                print("({}) [{}] detections".format(file_name, len(entry['annotations'])))
            else:
                print('False negative: [{}] - {}'.format(label, file_name))
                logfile.write('{}False negative: [{}] - {}'.format(nl, label, img))
                nl = "\n"

# Dump annotations and close open files
json.dump(detections, bbxfile)
bbxfile.close()
logfile.close()
