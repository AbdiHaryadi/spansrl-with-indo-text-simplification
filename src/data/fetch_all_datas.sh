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
if [ ! -e "srl_corpus.txt" ]; then
  gdown --id 1wE9wJM4v1nE4-1aSz2MudOhANkl4rc73
else
  echo "File srl_corpus.txt already exists."
fi

cd ".."

if [ ! -d $FEATURES_PATH ]; then
  mkdir -p $FEATURES_PATH
fi

cd $FEATURES_PATH

if [ ! -e "test_features.tar.gz" ]; then
  if [ ! -e "test_features.tar.gz.zip" ]; then
    echo "Downloading test features"
    gdown --id 1xBBVSMmmtX2lV0Bvv3-dKaHEJURowKNY
  else
    echo "File test_features.tar.gz.zip already exists."
  fi
  mv test_features.tar.gz.zip test_features.tar.gz
else
  echo "File test_features.tar.gz already exists."
fi

gzip -d test_features.tar.gz
tar -xf test_features.tar

cd ".."

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi

PRETRAINED_PATH="./pretrained"

if [ ! -d $PRETRAINED_PATH ]; then
  mkdir -p $PRETRAINED_PATH
fi

cd $PRETRAINED_PATH

if [ ! -e "word2vec_news.model.wv.vectors.zip" ]; then
  echo "Downloading pretrained model"
  gdown --id 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
else
  echo "File word2vec_news.model.wv.vectors.zip already exists."
fi

unzip word2vec_news.model.wv.vectors.zip
rm word2vec_news.model.wv.vectors.zip
cd ".."
cd ".."
