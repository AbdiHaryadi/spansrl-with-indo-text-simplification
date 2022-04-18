#! /bin/bash

MODEL_PATH="./models"
if [ ! -d $MODEL_PATH ]; then
  mkdir -p $MODEL_PATH
fi

RESULTS_PATH="./data/results"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi