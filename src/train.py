import sys
import json
import numpy as np
from .models import SRL
import tensorflow as tf

from .utils import eval_validation, load_data, print_default
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

def main():
    config = sys.argv[1]
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']

    # Multi GPU
    # devices = tf.config.experimental.list_physical_devices("GPU")
    # device_names = [d.name.split("e:")[1] for d in devices]
    # config_taken = [0,1,2,3]
    # taken_gpu = []
    # for i, device_name in enumerate(device_names):
    #     if i in config_taken:
    #         taken_gpu.append(device_name)
    # print(f"Taken GPU: {taken_gpu}")
    # strategy = tf.distribute.MirroredStrategy(devices=taken_gpu)

    # Temp code to handle multi GPU
    ## TO DO: REMOVE
    
    # Features loading
    input, out = load_data(config, 'train')
    input_val, out_val = load_data(config, 'val')

    # Training Parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    initial_learning_rate = config['learning_rate']

    print(len(input[0]))
    print(len(input_val[0]))
    # with strategy.scope():
    # with tf.device('/gpu:7'):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    print(config)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.999,
        staircase=True)

    
    # Define model
    model = SRL(config)
    # Checkpoint
    bestCheckpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'],
                                            save_best_only=True)
    lastCheckpoint = tf.keras.callbacks.ModelCheckpoint(config['checkpoint_path'],
                                            save_best_only=False)
    pruningCheckpoint = tf.keras.callbacks.ModelCheckpoint(config['model_path'],
                                            save_best_only=False)
    # Compiling, fitting and saving model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy())
    
    if (config['use_pruning']):
        model.fit(input, out, batch_size=batch_size, epochs=epochs, callbacks=[callback, pruningCheckpoint])
    else:
        model.fit(input, out, batch_size=batch_size, validation_data=(input_val, out_val), epochs=epochs, callbacks=[callback, bestCheckpoint, lastCheckpoint])

    with tf.device('/gpu:1'):
        eval_validation(config)

if __name__ == "__main__":
    main()