#!/bin/bash

cd craft_learn
nohup python craft_learn.py --dataset_type KO_MATH --op_mode PREPROCESS_ALL --ini_fname craft_learn_ko.ini > $(date +%y%m%d)_preprocess_ko_math.txt &
cd ..
