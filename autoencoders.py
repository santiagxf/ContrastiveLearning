import tensorflow as tf
import tensorflow.keras as keras


class Autoencoder(keras.Model):
    def __init__(self, encoding_dim = 10):
        super(Autoencoder, self).__init__()
        
        self.encode = keras.layers.Dense(encoding_dim, activation='relu')
        self.decode = keras.layers.Dense(29, activation='sigmoid')
    
    def call(self, inputs):
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        
        return decoded


