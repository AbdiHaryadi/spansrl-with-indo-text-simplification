import numpy as np
from ..features import SRLData
from tensorflow.keras.models import load_model

def create_span(length, max_span_length):
    span_start = []
    span_end = []
    span_width = []
    for i in range(length):
        for j in range(max_span_length):
            start_idx = i
            end_idx = i+j

            if (end_idx >= length):
                break
            span_start.append(start_idx)
            span_end.append(end_idx)
            span_width.append(end_idx - start_idx + 1)

    # Shape span_start, span_end: [num_spans]
    return span_start, span_end, span_width


def print_default():
    print('Configurations name not found.. Using the default config..')
            
def eval_test(config):
    input, out = load_data(config, 'test', False)

    # Predicting, unload model
    data = SRLData(config, emb=False)

    model = load_model(config['model_path'])
    if (config['use_pruning']):
        pred, idx_pred, idx_arg = model.predict(input, batch_size=config['batch_size'])
        res =  data.convert_result_to_readable(pred, idx_arg, idx_pred)
    else:
        pred = model.predict(input, batch_size=config['batch_size'])
        res =  data.convert_result_to_readable(pred)
    real = data.convert_result_to_readable(out)
    data.evaluate(real, res)
    cons = 'constraint' if config['use_constraint'] else ''
    with open('data/results/test_'+ cons + '_' + config['model_path'].split('/')[1]+'.txt', 'w') as f:
        for item in res:
            f.write("%s\n" %str(item))

def eval_validation(config):
    input, out = load_data(config, 'val', True)

    # Predicting, unload model
    data = SRLData(config, emb=False)

    model = load_model(config['model_path'])
    if (config['use_pruning']):
        pred, idx_pred, idx_arg = model.predict(input, batch_size=config['batch_size'])
        res =  data.convert_result_to_readable(pred, idx_arg, idx_pred)
    else:
        pred = model.predict(input, batch_size=config['batch_size'])
        res =  data.convert_result_to_readable(pred)
    real = data.convert_result_to_readable(out)
    data.evaluate(real, res)
    cons = 'constraint' if config['use_constraint'] else ''

    with open('data/results/val_'+ cons +'_' + config['model_path'].split('/')[1]+'.txt', 'w') as f:
        for item in res:
            f.write("%s\n" %str(item))

def load_data(config, types, eval=False):
     # Features loading
    dir = config['features_dir'] + types + '_'
    features_1 = np.load(dir+config['features_1'], mmap_mode='r')
    features_2 = np.load(dir+config['features_2'], mmap_mode='r')
    features_3 = np.load(dir+config['features_3'], mmap_mode='r')
   
    if (types == 'val' and not eval):
        n = 1
        input = [features_1[n:], features_2[n:], features_3[n:]]
        out = np.load(dir + config['output'], mmap_mode='r')[n:]
    else:
        input = [features_1, features_2, features_3]
        out = np.load(dir + config['output'], mmap_mode='r')
    
   
    return input, out