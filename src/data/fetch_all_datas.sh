#! /bin/bash

DATA_PATH="./data"
RAW_PATH="./raw"
FEATURES_PATH="./features"
RESULTS_PATH="./results"

if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

cd $DATA_PATH

if [ ! -d $RAW_PATH ]; then
  mkdir -p $RAW_PATH
fi

cd $RAW_PATH
gdown --id 1wE9wJM4v1nE4-1aSz2MudOhANkl4rc73

cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH

echo "Downloading features"
gdown --id 1-SmXIY2NuQLTLnl9qT0Abya3GMSHpilO
gdown --id 1cTc0An2FSXX-z3zod_UYA737NDh9gxR5
mv  val_features.tar.gz.zip val_features.tar.gz
mv  train_features.tar.gz.zip train_features.tar.gz
gzip -d train_features.tar.gz
gzip -d val_features.tar.gz
tar -xf train_features.tar
tar -xf val_features.tar

cd ".."

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi

PRETRAINED_PATH="./pretrained"

if [ ! -d $PRETRAINED_PATH ]; then
  mkdir -p $PRETRAINED_PATH
fi

cd $PRETRAINED_PATH

echo "Downloading pretrained models"
gdown --id 1553_9shAUrQpFAB0vqXbQrpHqJvWYgp4
unzip word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip
rm word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip
gdown --id 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
unzip word2vec_news.model.wv.vectors.zip
rm word2vec_news.model.wv.vectors.zip
cd ".."