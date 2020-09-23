#!/bin/bash

ROOT_DIR=$(pwd)
export PYTHONPATH=${ROOT_DIR}
echo

CRAFT_LEARN_DIR=${ROOT_DIR}/craft_learn
export PYTHONPATH=${PYTHONPATH}:${CRAFT_LEARN_DIR}
echo $PYTHONPATH

