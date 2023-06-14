@echo off

set DATA_PATH=".\data"
set RAW_PATH=".\raw"
set FEATURES_PATH=".\features"
set RESULTS_PATH=".\results"

if not exist %DATA_PATH% (
  mkdir %DATA_PATH%
)

cd %DATA_PATH%

if not exist %RAW_PATH% (
  mkdir %RAW_PATH%
)

cd %RAW_PATH%
if not exist "srl_corpus.txt" (
  gdown 1wE9wJM4v1nE4-1aSz2MudOhANkl4rc73
) else (
  echo File srl_corpus.txt already exists. 
)

cd ..

if not exist %FEATURES_PATH% (
  mkdir %FEATURES_PATH%
)

cd %FEATURES_PATH%

if not exist "test_features.tar.gz" (
  if not exist "test_features.tar.gz.zip" (
    echo Downloading test features
    gdown 1xBBVSMmmtX2lV0Bvv3-dKaHEJURowKNY
  ) else (
    echo File test_features.tar.gz.zip already exists.
  )
  ren test_features.tar.gz.zip test_features.tar.gz
) else (
  echo File test_features.tar.gz already exists.
)

cd ..

if not exist %RESULTS_PATH% (
  mkdir %RESULTS_PATH%
)

set PRETRAINED_PATH=".\pretrained"

if not exist %PRETRAINED_PATH% (
  mkdir %PRETRAINED_PATH%
)

cd %PRETRAINED_PATH%

if not exist "word2vec_news.model.wv.vectors.zip" (
  echo Downloading pretrained model
  gdown 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
) else (
  echo File word2vec_news.model.wv.vectors.zip already exists.
)

cd ..
cd ..

echo Please unarchive all archived files that has been downloaded.
