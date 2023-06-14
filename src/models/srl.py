from tensorflow.keras.models import Model

from tensorflow.keras.layers import Concatenate, Dropout, Dense, Embedding, Softmax, Input
from .blocks import Pruning, CustomAttention, BiHLSTM, SpanEndpointsLength, Scoring, ComputeScoring, CharacterEmbedding, PredicateArgumentEmb
from .layers import BiAffine
from ..utils import create_span

import tensorflow as tf

tf.random.set_seed(42)
class SRL(Model):

    def __init__(self, config, **kwargs):
        super(SRL, self).__init__(**kwargs)
        # Configurations and constants
        self.max_arg_span = config['max_arg_span']
        self.max_pred_span = config['max_pred_span']
        self.max_tokens = config['max_tokens']
        self.num_labels = len(config['srl_labels'])
        self.max_char = config['max_char']
        self.arg_span_idx = create_span(self.max_tokens, self.max_arg_span)
        self.pred_span_idx = create_span(self.max_tokens, self.max_pred_span)
        # Blocks and Layers
        self.character_block = CharacterEmbedding(config, name="character_embedding")
        # More embedding layer for wword emb and elmo?
        self.concatenate_1 = Concatenate(name="token_representation")
        self.dropout_1 = Dropout(config['emb_dropout'])

        self.bihlstm = BiHLSTM(config, name="BiLSTM")

        # MLP for predicate and argument (questionable)
        self.mlp_pred = Dense(config['mlp_pred_units'], activation='relu')
        self.mlp_arg = Dense(config['mlp_arg_units'], activation='relu')       

        # Span representation for argument 
        self.arg_endpoints_len = SpanEndpointsLength(self.arg_span_idx, name="arg_endpoints_and_length")
        self.arg_attention = CustomAttention(config, self.arg_span_idx, name="arg_attention_head")
        self.arg_length_emb = Embedding(input_dim=self.max_arg_span+1, output_dim=config['span_width_emb'], name='arg_width_emb')
        self.dropout_2 = Dropout(config['dropout_value'])
        self.concatenate_2 = Concatenate(name='arg_span_representation')

        # Span representation for predicate 
        if (self.max_pred_span > 1):
            self.pred_endpoints_len = SpanEndpointsLength(self.pred_span_idx, name="pred_endpoints_and_length")
            self.pred_attention = CustomAttention(config, self.pred_span_idx, name="pred_attention_head")
            self.pred_length_emb = Embedding(input_dim=self.max_pred_span+1, output_dim=config['max_pred_span'], name="pred_width_emb")
            self.dropout_3 = Dropout(config['dropout_value'])
            self.concatenate_3 = Concatenate(name='pred_span_representation')

        # Unary score
        self.pred_unary = Scoring(config, 1, name='pred_unary_score')
        self.arg_unary = Scoring(config, 1, name='arg_unary_score')
    

        # Pruning
        self.use_pruning = config['use_pruning']
        self.arg_span_idx_mask = []
        self.pred_span_idx_mask = []
        self.pred_prune = Pruning(config['max_predicates'], name='pred_pruning')
        self.arg_prune = Pruning(config['max_arguments'], name='arg_pruning')

        # Scoring for predicate-argument pair
        self.pred_arg_pair = PredicateArgumentEmb(name="pred_arg_pair")
        self.pred_arg_score = Scoring(config, self.num_labels, name="pred_arg_score")
        self.biaffine_score = BiAffine(self.num_labels, name='biaffine_relation_score')
        self.compute_score = ComputeScoring(self.num_labels, name='compute_final_score')

        self.softmax = Softmax(name="softmax_labels")


    def call(self, inputs, training=None): 
        # Shape word_emb: (batch_size, max_tokens, emb)
        # Shape elmo_emb: (batch_size, max_tokens, emb)
        # Shape char_input: (batch_size, max_tokens, max_char)
        word_emb, elmo_emb, char_input = inputs
        # Shape char_emb: (batch_size, max_tokens, char_output_dim)
        char_emb = self.character_block(char_input)

        # Shape token_rep: (batch_size, max_tokens, emb)
        token_rep = self.concatenate_1([word_emb, elmo_emb, char_emb])
        token_rep = self.dropout_1(token_rep)
        token_rep = self.bihlstm(token_rep)

        # MLP for predicate and argument
        mlp_pred_out = self.mlp_pred(token_rep) # Shape: (batch_size, max_tokens, mlp_units)
        mlp_arg_out = self.mlp_arg(token_rep) # Shape: (batch_size, max_tokens, mlp_units)

        # Span representation for argument
        # Shape arg_start_emb, arg_end_emb: (batch_size, num_spans, emb)
        # Shape arg_length: (batch_size, num_spans)
        arg_start_emb, arg_end_emb, arg_length = self.arg_endpoints_len(mlp_arg_out) 
        # Shape arg_head_emb, arg_width_emb: (batch_size, num_spans, emb)
        arg_head_emb = self.arg_attention(mlp_arg_out)
        arg_width_emb = self.arg_length_emb(arg_length)
        arg_width_emb = self.dropout_2(arg_width_emb)
        # Shape arg_rep: (batch_size, num_args, emb)
        arg_rep = self.concatenate_2([arg_start_emb, arg_end_emb, arg_head_emb, arg_width_emb])

        if (self.max_pred_span > 1):
            # Span representation for predicate
            # Shape pred_start_emb, pred_end_emb: (batch_size, num_spans, emb)
            # Shape pred_length: (batch_size, num_spans)
            pred_start_emb, pred_end_emb, pred_length = self.pred_endpoints_len(mlp_pred_out) 
            # Shape pred_head_emb, pred_width_emb: (batch_size, num_spans, emb)
            pred_head_emb = self.pred_attention(mlp_pred_out)
            pred_width_emb = self.pred_length_emb(pred_length)
            pred_width_emb = self.dropout_3(pred_width_emb)
            # Shape pred_rep: (batch_size, num_preds, emb)
            pred_rep = self.concatenate_3([pred_start_emb, pred_end_emb, pred_head_emb, pred_width_emb])
        else:
            # Shape pred_rep: (batch_size, num_preds, emb)
            pred_rep = mlp_pred_out

        # Unary score for predicate and argument
        # Shape pred_unary_score: (batch_size, num_preds, 1)
        pred_unary_score = self.pred_unary(pred_rep)
        # Shape arg_unary_score: (batch_size, num_args, 1)
        arg_unary_score = self.arg_unary(arg_rep)

        # Pruning in inference
        if (not training and self.use_pruning) :
            filtered_arg_idx = self.arg_prune(arg_unary_score)
            filtered_pred_idx  = self.pred_prune(pred_unary_score)
            self.arg_span_idx_mask = filtered_arg_idx
            self.pred_span_idx_mask = filtered_pred_idx
            arg_unary_score = tf.gather(arg_unary_score, filtered_arg_idx, axis=1, batch_dims=1)
            pred_unary_score = tf.gather(pred_unary_score, filtered_pred_idx, axis=1, batch_dims=1)
            # Shape arg_rep: (batch_size, max_args, emb)
            arg_rep = tf.gather(arg_rep, filtered_arg_idx, axis=1, batch_dims=1)
            # Shape pred_rep: (batch_size, max_preds, emb)
            pred_rep = tf.gather(pred_rep, filtered_pred_idx, axis=1, batch_dims=1)
          


        # Scoring for predicate-argument pair
        # Shape pred_arg_emb: (batch_size, num_preds, num_args, emb)
        pred_arg_emb = self.pred_arg_pair([arg_rep, pred_rep])
        # Shape pred_arg_emb: (batch_size, num_preds, num_args, emb)
        pred_arg_score = self.pred_arg_score(pred_arg_emb)
        relation_score = self.biaffine_score([pred_rep, arg_rep])

        # Shape final_score: (batch_size, num_preds, num_args, num_labels+1) 1= null label
        final_score = self.compute_score([arg_unary_score, pred_unary_score, pred_arg_score, relation_score])
        # Shape out: (batch_size, num_args, num_preds, num_labels+1) 1= null label
        out = self.softmax(final_score)
        if (training):
            return out
        else:
            if (self.use_pruning):
                return out, filtered_pred_idx, filtered_arg_idx
            else:
                return out

