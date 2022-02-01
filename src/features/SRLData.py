

import os
import sys
from collections import Counter

from pkg_resources import ExtractionError

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from helper import _print_f1, check_pred_id, split_first, label_encode, get_span_idx, pad_input, extract_bert, extract_pas_index, save_emb, convert_idx
from utils.utils import create_span
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from gensim.models import fasttext, Word2Vec
import time


class SRLData(object):
    def __init__(self, config):
        self.config = config
        # Configurations and constants
        self.max_arg_span = config['max_arg_span']
        self.max_pred_span = config['max_pred_span']
        self.max_tokens = config['max_tokens']
        self.num_labels = len(config['srl_labels'])
        self.max_char = config['max_char']
        self.arg_span_idx = create_span(self.max_tokens, self.max_arg_span)
        self.pred_span_idx = create_span(self.max_tokens, self.max_pred_span)

        # Initial Data
        self.sentences = []
        self.arg_list = []

        # Features Input
        ## Character
        initial_char_dict = {"<pad>":0, "<unk>":1}
        self.char_dict = label_encode(config['char_list'], initial_char_dict)
        self.char_input = []

        ## Word Embedding
        self.use_fasttext = config['use_fasttext']
        self.padded_sent = []
        self.word_vec = Word2Vec.load(config['word_emb_path']).wv
        self.word_emb = []
        self.emb1_dim = 300
        if (config['use_fasttext']):
            self.fast_text = fasttext.load_facebook_vectors(config['fasttext_emb_path'])
            self.word_emb_2 = []
            self.emb2_dim = 300
        else:
            self.bert_model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
            self.bert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1", padding_side='right')
            self.word_emb_2 = []
            self.emb2_dim = 768
        # Output
        self.output = []

        # Dict
        self.labels_mapping = label_encode(config['srl_labels'])


    # To convert train data labels to output models
    def convert_train_output(self):
        batch_size = len(self.sentences)
        pred_start, _, _ = self.pred_span_idx
        arg_start, _, _ = self.arg_span_idx

        num_preds = len(pred_start)
        num_args = len(arg_start)
        # Fill with null labels first
        print('trying')
        initialData = np.zeros([batch_size, num_preds, num_args, self.num_labels])
        print('trying 2')
        initialLabel = np.ones([batch_size, num_preds, num_args, 1])
        print('trying 3')
        initialData = np.concatenate([initialData, initialLabel], axis=-1)
        print('trying 4')
        indices = []
        indices_null = []
        for idx_sent, sentences in tqdm(enumerate(self.arg_list)):
            for PAS in sentences:
                id_pred_span = get_span_idx(PAS['id_pred'], self.pred_span_idx)
                if (id_pred_span == -1):
                    continue
                arg_list = PAS['args']
                for arg in arg_list:
                    arg_idx = arg[:2]
                    id_arg_span = get_span_idx(arg_idx, self.arg_span_idx)
                    if (id_arg_span == -1):
                        continue
                    try :
                        label_id = self.labels_mapping[arg[-1]]
                    except:
                        print(arg[-1])
                        continue
                    indice_pas = [idx_sent, id_pred_span, id_arg_span, label_id]
                    # max length = num_labels + 1
                    indice_reset = [idx_sent, id_pred_span, id_arg_span, self.num_labels]
                    indices.append(indice_pas)
                    indices_null.append(indice_reset)
        updates = [1 for _ in indices]
        updates_null = [0 for _ in indices_null]
        initialData =  tf.tensor_scatter_nd_update(initialData, indices_null, updates_null)
        self.output = tf.tensor_scatter_nd_update(initialData, indices, updates)
        save_emb(self.output, "train", "output")
        print(self.output.shape)

    def extract_features(self, type, isSum=False):
        self.pad_sentences(isArray=isSum and type == 'test')
        if (isSum):
            # berishin dulu
            cleaned_sent = []
                # Documents
            self.word_emb = [self.extract_ft_emb(sent, padded) for sent, padded in zip(cleaned_sent, self.padded_sent)]   
            self.word_emb_2 = [self.extract_sec_emb(sent) for sent in cleaned_sent]
            self.char_input = [self.extract_char(sent) for sent in cleaned_sent]
        else:
            print('extracting word emb 2 features')
            self.word_emb_2 = self.extract_word_emb(self.sentences, self.padded_sent)
            print('extracting word emb features')
            self.word_emb = self.extract_ft_emb(self.sentences, self.padded_sent)
            print('extracting char features')
            self.char_input = self.extract_char(self.padded_sent)


        save_emb(self.word_emb, 'word_emb', type, isSum)
        if (self.use_fasttext):
            name = 'fasttext'
        else:
            name = 'bert'
        save_emb(self.word_emb_2, name, type, isSum)
        save_emb(self.char_input, 'char_input', type, isSum)
        print(self.word_emb.shape)
        print(self.word_emb_2.shape)
        print(self.char_input.shape)
    
    def extract_word_emb(self, sentences, padded_sent):
        word_emb = np.zeros(shape=(len(sentences), self.max_tokens, 300))
        for i, sent in tqdm(enumerate(padded_sent)):
            for j, word in enumerate(sent):
                if (word == '<pad>' or not self.word_vec.has_index_for(word.lower())):
                    continue
                word_vec = self.word_vec[word.lower()]
                word_emb[i][j] = word_vec
        return word_emb        

    def extract_sec_emb(self, sentences):
        if (self.use_fasttext):
            return self.extract_ft_emb(sentences, self.padded_sent)
        else:
            return self.extract_bert_emb(sentences)
    def extract_bert_emb(self, sentences): # sentences : Array (sent)
        bert_emb = extract_bert(self.bert_model, self.bert_tokenizer, sentences, self.max_tokens)
        bert_emb = np.array(bert_emb)
        return bert_emb
        

    def extract_ft_emb(self, sentences, padded_sent):  # sentences : Array (sent)
        word_emb = np.ones(shape=(len(sentences), self.max_tokens, 300))
        for i, sent in tqdm(enumerate(padded_sent)):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    word_vec = np.zeros(300)
                else:
                    word_vec = self.fast_text[word.lower()]
                word_emb[i][j] = word_vec
        return word_emb
    
    def extract_char(self, sentences): # sentences: Array (sent)
        char = np.zeros(shape=(len(sentences), self.max_tokens, self.max_char))
        for i, sent in tqdm(enumerate(sentences)):
            for j, word in enumerate(sent):
                if (word == '<pad>'):
                    continue
                char_encoded = [self.char_dict[x]  if x in self.char_dict else self.char_dict['<unk>'] for x in word]
                if (len(char_encoded) >= self.max_char):
                    char[i][j] = char_encoded[:self.max_char]
                else:
                    char[i][j][:len(char_encoded)] = char_encoded
        return char

    def pad_sentences(self, isArray=False):
        if (isArray):
            padded = [pad_input(sent, self.max_tokens) for sent in self.sentences]
        else:
            padded = pad_input(self.sentences, self.max_tokens)
        self.padded_sent = padded

    def convert_result_to_readable(self, out, arg_mask=None, pred_mask=None): # (batch_size, num_preds, num_args, num_labels)
        labels_list = list(self.labels_mapping.keys())
        transpose = tf.transpose(out, [3, 0, 1, 2])
        omit = tf.transpose(transpose[:-1], [1, 2, 3, 0])
        #array of position
        ids = tf.where(omit)
        pas = convert_idx(ids, len(out), self.arg_span_idx, self.pred_span_idx, labels_list, arg_mask, pred_mask)
        return pas

    def evaluate(self, y, pred):
        # Adopted from unisrl
        total_gold = 0
        total_pred = 0
        total_matched = 0
        total_unlabeled_matched = 0
        comp_sents = 0
        label_confusions = Counter()

        for y_sent, pred_sent in zip(y, pred):
            gold_rels = 0
            pred_rels = 0
            matched = 0
            for gold_pas in y_sent:
                pred_id = gold_pas['id_pred']
                gold_args = gold_pas['args']
                total_gold += len(gold_args)
                gold_rels += len(gold_args)
                
                arg_list_in_predict = check_pred_id(pred_id, pred_sent)
                if (len(arg_list_in_predict) == 0):
                    continue
                for arg0 in gold_args:
                    for arg1 in arg_list_in_predict[0]:
                        if (arg0[:-1] == arg1[:-1]): # Right span
                            total_unlabeled_matched += 1
                            label_confusions.update([(arg0[2], arg1[2]),])
                            if (arg0[2] == arg1[2]): # Right label
                                total_matched += 1
                                matched += 1
            for pred_pas in pred_sent:
                pred_id = pred_pas['id_pred']
                pred_args = pred_pas['args']
                total_pred += len(pred_args)
                pred_rels += len(pred_args)
            
            if (gold_rels == matched and pred_rels == matched):
                comp_sents += 1

        precision, recall, f1 = _print_f1(total_gold, total_pred, total_matched, "SRL")
        ul_prec, ul_recall, ul_f1 = _print_f1(total_gold, total_pred, total_unlabeled_matched, "Unlabeled SRL")
        
        return


    def read_raw_data(self):
        file = open(os.getcwd() + self.config['train_data'], 'r')
        lines = file.readlines()
        max = 0
        for pairs in lines:
            # Split sent, label list
            sent, PASList = split_first(pairs, ';')
            tokens = sent.split(' ')
            arg_list, max, sent = extract_pas_index(PASList, tokens, self.max_tokens, max)

            # Append for differents sentences
            self.sentences.append(tokens)
            self.arg_list.append(arg_list)
        print('max args = '+ str(max))
        print('di sentence '+ str(sent))

        
