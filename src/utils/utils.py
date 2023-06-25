from itertools import permutations
import time
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
    timestamp = int(time.time() // 1)
    _, out = load_test_data(config, False)

    # Predicting, unload model
    data = SRLData(config, emb=False)
    unsimplified_sentences, _ = data.read_raw_data()

    output_pas_list_collection = []
    if config["use_simplification"]:
        # Handle simplification map
        sentence_indices_list = []
        with open(config['features_dir'] + "simplification_map.txt", mode="r") as file:
            for line in file.readlines():
                sentence_indices_list.append(list(map(int, line.strip().split())))

        with open("data/features/simplified_sentences.txt", mode="r") as f:
            simplified_sentences = f.readlines()

        res = get_prediction_result(config, result_path=f"{timestamp}_result.txt")

        print("res:", res)
        for indices in sentence_indices_list:
            output_pas_list = []
            print("indices:", indices)
            for i in indices:
                print("i:", i)
                result = res[i]
                word_list = simplified_sentences[i].strip().split(" ")
                for r in result:
                    pred_s, pred_e = r['id_pred']
                    predicate = ' '.join(word_list[pred_s:pred_e+1]).lower()

                    arguments = []
                    for arg in r['args']:
                        s, e, label = arg
                        argument_text = ' '.join(word_list[s:e+1]).lower()
                        arguments.append((argument_text, label))

                    output_pas_list.append({ "predicate": predicate, "arguments": arguments })

            output_pas_list_collection.append(output_pas_list)

    else:
        res = get_prediction_result(config, result_path=f"{timestamp}_result.txt")
        for i, result in enumerate(res):
            output_pas_list = []
            for r in result:
                pred_s, pred_e = r['id_pred']
                predicate = ' '.join(unsimplified_sentences[i][pred_s:pred_e+1]).lower()

                arguments = []
                for arg in r['args']:
                    s, e, label = arg
                    argument_text = ' '.join(unsimplified_sentences[i][s:e+1]).lower()
                    arguments.append((argument_text, label))

                output_pas_list.append({ "predicate": predicate, "arguments": arguments })

            output_pas_list_collection.append(output_pas_list)

    real = data.convert_result_to_readable(out)

    print("Evaluating ...")

    target_pas_list_collection = []
    for i, result in enumerate(real):
        target_pas_list = []
        for r in result:
            pred_s, pred_e = r['id_pred']
            predicate = ' '.join(unsimplified_sentences[i][pred_s:pred_e+1]).lower()

            arguments = []
            for arg in r['args']:
                s, e, label = arg
                word_list = unsimplified_sentences[i][s:e+1]
                argument_text = ' '.join(word_list).lower() # Tidak menggunakan word_list_to_sentence karena bukan kalimat.
                arguments.append((argument_text, label))

            target_pas_list.append({ "predicate": predicate, "arguments": arguments })

        target_pas_list_collection.append(target_pas_list)

    index = 0
    for output_pas_list, target_pas_list  in zip(output_pas_list_collection, target_pas_list_collection):
        result = direct_eval(output_pas_list, target_pas_list)
        with open(f"{timestamp}_result.txt", mode="a") as file:
            print("---", file=file)
            print("Index:", index, file=file)
            print("Output:", output_pas_list, file=file)
            print("Target:", target_pas_list, file=file)
            print("Result:", result, file=file)
            print("---", file=file)

        index += 1

def get_prediction_result(config, result_path="result.txt"):
    input_feats, _ = load_test_data(config, False)
    data = SRLData(config, emb=False)

    res = []
    model = load_model(config['model_path'])
    dataset_size = len(input_feats[0])
    start_time = time.time()
    current_duration = 0.0
    for index in range(dataset_size):
        single_input_feats = [feat[index:index+1, :, :] for feat in input_feats]

        if (config['use_pruning']):
            pred, idx_pred, idx_arg = model.predict(single_input_feats, batch_size=config['batch_size'])
            single_res =  data.convert_result_to_readable(pred, idx_arg, idx_pred)
        else:
            pred = model.predict(single_input_feats, batch_size=config['batch_size'])
            single_res = data.convert_result_to_readable(pred)

        res += single_res
        current_duration = time.time() - start_time
        
        with open(result_path, mode="a") as file:
            print(single_res[0], file=file)

        print("{}/{}\t({:.2f} s/{:.2f} s)".format(index + 1, dataset_size, current_duration, current_duration + current_duration * (dataset_size - 1 - index) / (index + 1)))
    return res

def load_test_data(config, eval=False):
     # Features loading
    dir = config['features_dir'] + "test" + '_'
    input_dir = dir + ("simplified_" if config["use_simplification"] else "")
    features_1 = np.load(input_dir+config['features_1'], mmap_mode='r')
    features_2 = np.load(input_dir+config['features_2'], mmap_mode='r')
    features_3 = np.load(input_dir+config['features_3'], mmap_mode='r')
   
    input = [features_1, features_2, features_3]
    out = np.load(dir + config['output'], mmap_mode='r')
   
    return input, out

def direct_eval(output_pas_list: list[dict[str, str]], target_pas_list: list[dict[str, str]]):
    output_pas_argument_count = sum([len(pas["arguments"]) for pas in output_pas_list])
    target_pas_argument_count = sum([len(pas["arguments"]) for pas in target_pas_list])
    
    output_pas_count = len(output_pas_list)
    target_pas_count = len(target_pas_list)

    match = -1
    if target_pas_count <= output_pas_count:
        output_pas_list_indices = list(range(output_pas_count))
        for permutated_output_pas_list_indices in permutations(output_pas_list_indices, target_pas_count):
            permutated_output_pas_list = [output_pas_list[index] for index in permutated_output_pas_list_indices]
            candidate_match = get_candidate_match_by_pas_list(permutated_output_pas_list, target_pas_list)

            if candidate_match > match:
                match = candidate_match

    else:
        target_pas_list_indices = list(range(output_pas_count))
        for permutated_target_pas_list_indices in permutations(target_pas_list_indices, output_pas_count):
            permutated_target_pas_list = [target_pas_list[index] for index in permutated_target_pas_list_indices]
            candidate_match = get_candidate_match_by_pas_list(output_pas_list, permutated_target_pas_list)

            if candidate_match > match:
                match = candidate_match
                
    precision = match / output_pas_argument_count if output_pas_argument_count != 0.0 else 0.0
    recall = match / target_pas_argument_count if target_pas_argument_count != 0.0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0.0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def get_candidate_match_by_pas_list(output_pas_list, target_pas_list):
    candidate_match = 0
    for output_pas, target_pas in zip(output_pas_list, target_pas_list):
        if target_pas["predicate"] == output_pas["predicate"]:
            candidate_match += get_candidate_match(output_pas, target_pas)
    return candidate_match

def get_candidate_match(output_pas, target_pas):
    current_total_candidate_match = 0
    for target_arg in set(target_pas["arguments"]):
        target_arg_count = len([arg for arg in target_pas["arguments"] if arg == target_arg])
        output_arg_count = len([arg for arg in output_pas["arguments"] if arg == target_arg])
        current_total_candidate_match += min(target_arg_count, output_arg_count)
    return current_total_candidate_match

def detailed_direct_eval(output_pas_list: list[dict[str, str]], target_pas_list: list[dict[str, str]]):
    output_detail = get_pas_detail(output_pas_list)
    target_detail = get_pas_detail(target_pas_list)

    output_pas_count = len(output_pas_list)
    target_pas_count = len(target_pas_list)

    match = -1
    current_match_detail = {
        "argument": {
            "all": 0
        },
        "predicate": 0
    }

    if target_pas_count <= output_pas_count:
        output_pas_list_indices = list(range(output_pas_count))
        for permutated_output_pas_list_indices in permutations(output_pas_list_indices, target_pas_count):
            permutated_output_pas_list = [output_pas_list[index] for index in permutated_output_pas_list_indices]
            candidate_detail = get_detailed_match_by_pas_list(permutated_output_pas_list, target_pas_list)
            candidate_match: int = candidate_detail["argument"]["all"]

            if candidate_match > match:
                match = candidate_match
                current_match_detail = candidate_detail

    else:
        target_pas_list_indices = list(range(output_pas_count))
        for permutated_target_pas_list_indices in permutations(target_pas_list_indices, output_pas_count):
            permutated_target_pas_list = [target_pas_list[index] for index in permutated_target_pas_list_indices]
            candidate_detail = get_detailed_match_by_pas_list(output_pas_list, permutated_target_pas_list)
            candidate_match: int = candidate_detail["argument"]["all"]

            if candidate_match > match:
                match = candidate_match
                current_match_detail = candidate_detail
                
    return {
        "match": current_match_detail,
        "output": output_detail,
        "target": target_detail
    }

def get_pas_detail(pas_list):
    detail = {
        "argument": {
            "all": 0
        },
        "predicate": 0
    }
    for pas in pas_list:
        detail["predicate"] += 1
        for arg in set(pas["arguments"]):
            detail["argument"]["all"] += 1
            detail["argument"][arg[1]] = detail["argument"].get(arg[1], 0) + 1
    return detail

def get_detailed_match_by_pas_list(output_pas_list, target_pas_list):
    total_detailed_argument_match_result: dict[str, int] = {
        "all": 0
    }
    predicate_match_result = 0

    for output_pas, target_pas in zip(output_pas_list, target_pas_list):
        if target_pas["predicate"] == output_pas["predicate"]:
            predicate_match_result += 1
            detailed_match_result = get_detailed_match(output_pas, target_pas)
            for scope, value in detailed_match_result.items():
                total_detailed_argument_match_result[scope] = total_detailed_argument_match_result.get(scope, 0) + value
    
    return {
        "argument": total_detailed_argument_match_result,
        "predicate": predicate_match_result
    }

def get_detailed_match(output_pas, target_pas):
    detailed_match_result: dict[str, int] = {
        "all": 0
    }

    for target_arg in set(target_pas["arguments"]):
        target_arg_count = len([arg for arg in target_pas["arguments"] if is_argument_same(arg, target_arg)])
        output_arg_count = len([arg for arg in output_pas["arguments"] if is_argument_same(arg, target_arg)])
        match_count = min(target_arg_count, output_arg_count)

        detailed_match_result[target_arg[1]] = detailed_match_result.get(target_arg[1], 0) + match_count
        detailed_match_result["all"] += match_count

    return detailed_match_result

def is_argument_same(arg_a: tuple[str, str], arg_b: tuple[str, str]) -> bool:
    if arg_a[1] == arg_b[1]:
        return arg_a[0].replace(" ", "") == arg_b[0].replace(" ", "")
    else:
        return False
