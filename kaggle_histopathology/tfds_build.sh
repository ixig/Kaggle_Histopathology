#!/bin/bash

if [ ! -f kaggle_histopathology.py ]; then
    echo "Execute this script in the directory created with 'tfds new ...'"
    exit 1
fi

rm -rf ~/tensorflow_datasets/kaggle_histopathology/

tfds build

CURR_DIR=`pwd`
rm $CURR_DIR/data.zip
pushd ~/tensorflow_datasets/kaggle_histopathology/ && zip -r $CURR_DIR/data.zip * && popd
