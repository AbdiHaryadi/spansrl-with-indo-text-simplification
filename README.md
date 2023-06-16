# Span Based Semantic Role Labeling with Text Simplification
This repository is based on [https://github.com/feliciagojali/spansrl](https://github.com/feliciagojali/spansrl), with text simplification integration. Note that this has been tested only on Windows.

# Preparation
Make sure Python 3 has been installed. Then, install the dependencies with this command:
```
pip install -r requirements.txt
```

Run this first:
```
prepare
```

Then, for fetching datasets used to train SRL models in this repo, run this command:
```
fetch_data
```
The fetched datas include srl data in particular format (raw) and its features to prepare for training.
This command result also includes Word2Vec model.

For pretrained predictor model, download from [https://drive.google.com/drive/u/1/folders/16za_WTXurgTzpQc9cXZc9NvHEyZYWznS](https://drive.google.com/drive/u/1/folders/16za_WTXurgTzpQc9cXZc9NvHEyZYWznS). Put it to `models` folder in root.

# Usage
Run predict script. 
```
predict <config>
```
For <config>, use `default` value to predict with text simplification. If text simplification don't want to be used, use `unsimplified` value.

# Note
SRL training here is not supported. Please go to the base of this repository.
