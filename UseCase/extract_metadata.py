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
import csv

CSVFILE = 'nacti_metadata_mini.csv'

# Open metadata file
csvfile = open(CSVFILE, 'r')
reader = csv.reader(csvfile)
# Burn first row
next(reader)
# Loop through entries looking for CPW data
images = {}
for row in reader:
    # column 4 is study, column 3 is filname, column 14 is common name label
    if row[3] == 'CPW':
        if row[2] not in images:
            images[row[2]] = row[13]
        else:
            # If the image name already exists and has different label delete
            if row[13] != images[row[2]]:
                del images[row[2]]
                print("Detected duplicate entry with different labels: {}".format(row[2]))
csvfile.close()

# Create new csv file with only the needed data
csvfile = open('cpw_labelmap.csv', 'w')
nl = ''
for key in images:
    csvfile.write("{}{},{}".format(nl, key, images[key]))
    nl = "\n"
csvfile.close()
