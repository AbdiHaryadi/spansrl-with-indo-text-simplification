raw_path = 'data/raw/'
processed_path = 'data/processed/'
raw_filename = 'raw.csv'

# import tensorflow as tf
import json
import numpy as np
from SRLData import SRLData
from helper import split_into_batch
import ast
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
def main():
    filename = './configurations.json'
    # path = 'data/raw/'
    # types = ['val']
    f = open(filename)
    all_config = json.load(f)

    data = SRLData(all_config['summary'])
    # path = 'data/processed/'
    # file = path+'test_summary_corpus.csv'

    # datas = pd.read_csv(file)

    # datas['article'] = datas['article'].progress_apply(lambda x : ast.literal_eval(x))
    # for i in datas:
    #     data.extract_features(i, 'val', True)
   
    # data.read_raw_data()
    data.extract_features("train", True)
    # data.convert_train_output()
        
   


if __name__ == "__main__":
    main()