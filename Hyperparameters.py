import tensorflow as tf
parameters = {
    "conv": {"kernelSize": [3, 5, 7], #11],
             "kernelCount": [32, 64, 96, 128, 160, 192, 224, 256],#, 1024],
             "padding": ["SAME"],
             "dropoutRate": [0.3, 0.4, 0.5]},
    "pool": {"kernelSize": [2, 3],
             "dropoutRate": [0.3, 0.4, 0.5],
             "poolType": ["MAX", "AVG"]},
    "fullyConnected": {"unitCount": [128, 256, 512], #, 1024],
                       "dropoutRate": [0.3, 0.4, 0.5]},
    "learningProcess": {"activation": ['relu', 'leaky_relu', 'elu'], #tf.keras.layers.LeakyReLU()],# tf.keras.layers.PReLU()
                        "learningRate": [0.0001, 0.001, 0.01],
                        "batchSize": [64, 128, 256]},
    "seedNumber": 10,
    "ratioInit": 0.9
}
