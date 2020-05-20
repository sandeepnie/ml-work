# -*- coding: utf-8 -*-

#!/usr/bin/python

import sys
import os

var = str(sys.argv[2])

if var == 'train':
    print("Starting the Training of the Model ...")
    os.system("python -m src.train "+ str(sys.argv[1]))
if var == 'predict':
    print("Running the prediction module ...")
    os.system("python -m src.predict "+ str(sys.argv[1]))