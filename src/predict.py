import sys
import json
import nltk
import tensorflow as tf
from .features import SRLData
from nltk.tokenize import word_tokenize
from .utils import print_default
from tensorflow.keras.models import load_model

nltk.download('punkt')

config = sys.argv[1]
filename = './configurations.json'
f = open(filename)
all_config = json.load(f)

from indo_ts.src.indo_ts import TextSimplifier

with tf.device('/gpu:4'):
    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']

    srl_data = SRLData(config)
    srl_model = load_model(config['model_path'])

    simplifier: TextSimplifier | None = None
    if config["use_simplification"]:
        simplifier = TextSimplifier(tokenize_no_ssplit=True)
    # else: keep as None

    print('Input your own sentence')
    print('---------------------------')
    print('Please input one sentences at a time and please end with `END` if already done')
    print('Example, you want to inference 3 sentences, therefore type `Sentence 4: END`')
    
    while(True):
        raw_sentences: list[str] = []
        while(True):
            sent = str(input('Sentence ' + str(len(raw_sentences) + 1) +' : '))
            if sent != 'END':
                if (sent == ''):
                    continue
                raw_sentences.append(sent)
            else:
                break
        if (len(raw_sentences) == 0):
            print('No sentence detected. Exit.')
            break

        
        if simplifier is None:
            sentences: list[str] = []
            for sentence in raw_sentences:
                tokenized = word_tokenize(sentence)
                sentences.append(tokenized)

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

        else:
            print('Simplifying sentences...')
            modified_raw_sentences: list[str] = []
            for sentence in raw_sentences:
                modified_raw_sentences.append(sentence.strip())
            raw_sentences = modified_raw_sentences

            simplified_sentences_list: list[list[str]] = simplifier.simplify(raw_sentences)
            sentences: list[str] = []
            simplification_indices_list: list[list[int]] = []

            print("Simplification result:")
            index = 0
            for original_sentence_index, simplified_sentences in enumerate(simplified_sentences_list):
                print(raw_sentences[original_sentence_index])
                current_simplification_indices: list[int] = []
                for sentence in simplified_sentences:
                    print("-", sentence)
                    tokenized = word_tokenize(sentence)
                    sentences.append(tokenized)

                    current_simplification_indices.append(index)
                    index += 1

                simplification_indices_list.append(current_simplification_indices)

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

            for original_sentence_index, simplification_indices in enumerate(simplification_indices_list):
                print('Sentence '+ str(original_sentence_index+1) +'  :')
                print('===')
                for simplification_index, sentence_index in enumerate(simplification_indices):
                    print(f"{simplification_index + 1}:")
                    result = res[sentence_index]
                    for r in result:
                        print(str(r))
                        pred_s, pred_e = r['id_pred']
                        print('Predikat: ' + ' '.join(sentences[sentence_index][pred_s:pred_e+1]))
                        print('Argumen:')
                        for arg in r['args']:
                            s, e, label = arg
                            print(label + ': ' + ' '.join(sentences[sentence_index][s:e+1]))
                        print('===')
                    
                print('-----------')
