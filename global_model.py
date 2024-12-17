import tensorflow as tf
import tensorflow_federated as ttf

def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(784,))
    ])

    model.compile(
        loss = tf.keras.losses.SparseCategoricalEntropy(),
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

