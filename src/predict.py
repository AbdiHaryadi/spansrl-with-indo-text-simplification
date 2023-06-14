import sys
import json
import nltk
import tensorflow as tf
from .features import SRLData
from nltk.tokenize import word_tokenize
from .utils import eval_validation, eval_test, print_default
from tensorflow.keras.models import load_model

nltk.download('punkt')

config = sys.argv[1]
filename = './configurations.json'
f = open(filename)
all_config = json.load(f)



with tf.device('/gpu:4'):
    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']

    srl_data = SRLData(config)
    srl_model = load_model(config['model_path'])

    print('Input your own sentence')
    print('---------------------------')
    print('Please input one sentences at a time and please end with `END` if already done')
    print('Example, you want to inference 3 sentences, therefore type `Sentence 4: END`')
    while(True):
        sentences = []
        while(True):
            sent = str(input('Sentence ' + str(len(sentences) + 1) +' : '))
            if sent != 'END':
                if (sent == ''):
                    continue
                tokenized = word_tokenize(sent)
                sentences.append(tokenized)
            else:
                break
        if (len(sentences) == 0):
            print('Please enter at least one sentences!')
            continue
    
        print('Extracting features...')

        srl_data.extract_features(sentences)
        input_feat = [srl_data.word_emb_w2v, srl_data.word_emb_2, srl_data.char_input]


        if (config['use_pruning']):
            pred, idx_pred, idx_arg = srl_model.predict(input_feat, batch_size=config['batch_size'])
            res =  srl_data.convert_result_to_readable(pred, idx_arg, idx_pred)
        else:
            pred = srl_model.predict(input_feat, batch_size=config['batch_size'])
            res =  srl_data.convert_result_to_readable(pred)
        
        print('Result')
        print('-----------')
        for i ,result in enumerate(res):
            print('Sentence '+ str(i+1) +'  :')
            print('===')
            for r in result:
                print(str(r))
                pred_s, pred_e = r['id_pred']
                print('Predikat: ' + ' '.join(sentences[i][pred_s:pred_e+1]))
                print('Argumen:')
                for arg in r['args']:
                    s, e, label = arg
                    print(label + ': ' + ' '.join(sentences[i][s:e+1]))
                print('===')
                
            print('-----------')