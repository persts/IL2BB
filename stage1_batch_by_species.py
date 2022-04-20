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
import shutil

# Little error checking to get things started
if len(sys.argv) == 1:
    print('Usage: python3 {} metadata data_dir batch_dir'.format(sys.argv[0]))
    sys.exit(0)

METADATA = sys.argv[1]
DATA = sys.argv[2]
BATCH = sys.argv[3]
if not os.path.isdir(DATA):
    print("[{}] is not valid directory.".format(DATA))
    sys.exit(0)

# open metadata file
csvfile = open(METADATA, 'r')
reader = csv.reader(csvfile)

# load data into dictionary
base = {}
for row in reader:
    if row[1] not in base:
        base[row[1]] = []
    base[row[1]].append(row[0])
csvfile.close()

# Copy and rename data in batches of 1000 by label
for label in base.keys():
    print(label)
    file_list = base[label]
    counter = 1000
    batch = 0
    current_batch = ''
    for src_file_name in file_list:
        if counter == 1000:
            counter = 0
            batch += 1
            # Set the current batch, make directory and open new csv file no error checking
            print("Batch [{}]".format(batch))
            batch_root = os.path.join(BATCH, label)
            current_batch = os.path.join(batch_root, "batch_{:03}".format(batch)) 
            os.makedirs(current_batch)
            mapfile = open(os.path.join(current_batch, "labels.csv"), 'w')
            nl = ''
        # copy the original image
        current_file = os.path.join(DATA, src_file_name)
        new_file_name = "img_{:03}.{}".format(counter, src_file_name[-3:].lower())
        new_file = os.path.join(current_batch, new_file_name)
        shutil.copy2(current_file, new_file) 
        mapfile.write("{}{},{},{}".format(nl, new_file_name, label, src_file_name))
        counter += 1
        nl = "\n"
        if counter == 1000:
            mapfile.close()
