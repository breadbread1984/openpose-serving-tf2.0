#!/usr/bin/python3

import tensorflow as tf;

def main():

    tf.keras.backend.set_learning_phase(0);
    model = tf.keras.models.load_model('model.h5');
    tf.saved_model.save(model, './openpose/1');
    loaded = tf.saved_model.load('./openpose/1');
    infer = loaded.signatures['serving_default'];
    print('output tensor name is', infer.structured_outputs);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
