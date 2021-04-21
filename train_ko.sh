#!/bin/bash

cd craft_learn
nohup python craft_learn.py --dataset_type KO --op_mode TRAIN --ini_fname craft_learn_ko.ini > $(date +%y%m%d)_train_ko.txt &
cd ..
