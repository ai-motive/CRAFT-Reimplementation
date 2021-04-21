#!/bin/bash

cd craft_learn
nohup python craft_learn.py --dataset_type MATH --op_mode TRAIN --ini_fname craft_learn_math.ini > $(date +%y%m%d)_train_math.txt &
cd ..
