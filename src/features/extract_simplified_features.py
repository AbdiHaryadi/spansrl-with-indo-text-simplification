import sys
import json
import numpy as np
import tensorflow as tf


from .helper import SRLData
from ..utils import print_default

from indo_ts.src.indo_ts import TextSimplifier
from indo_ts.src.utils import sentence_to_word_list, word_list_to_sentence

def main():
    config = sys.argv[1]
    filename = './configurations.json'
    with open(filename) as f:
        all_config = json.load(f)

    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']
    
    srl_data = SRLData(config)

    # Read raw data to get sentences and argument list
    sentences, _ = srl_data.read_raw_data()

    simplifier = TextSimplifier() # TODO: What is this?
    new_sentences = []
    simplification_map = []

    print("Simplifying ....")
    simplification_index = 0
    for word_list in sentences:
        current_simplification_indices = []
        simplified_sent_result_list = simplifier.simplify(word_list_to_sentence(word_list))
        for simplified_sent_result in simplified_sent_result_list:
            for simplified_sent in simplified_sent_result:
                word_list = sentence_to_word_list(simplified_sent)
                new_sentences.append(word_list)
                current_simplification_indices.append(simplification_index)
                simplification_index += 1

        simplification_map.append(current_simplification_indices)
        
    sentences = new_sentences
    print("Sentence count:", simplification_index)      

    # Extract features from sentences
    srl_data.extract_features(sentences)
    
    features_1 = srl_data.word_emb_w2v
    features_2 = srl_data.word_emb_2
    features_3 = srl_data.char_input

    print("Saving ...")
    dir = config['features_dir']
    np.save(dir + f"test_simplified_{config['features_1']}", features_1)
    np.save(dir + f"test_simplified_{config['features_2']}", features_2)
    np.save(dir + f"test_simplified_{config['features_3']}", features_3)

    print("Almost done ...")
    with open(dir + "simplified_sentences.txt", mode="w") as file:
        for word_list in sentences:
            print(*word_list, sep=" ", file=file)

    with open(dir + "simplification_map.txt", mode="w") as file:
        for simplification_indices in simplification_map:
            print(*simplification_indices, sep=" ", file=file)

    print("Done!")

if __name__ == "__main__":
    with tf.device('/gpu:3'):
        main()
