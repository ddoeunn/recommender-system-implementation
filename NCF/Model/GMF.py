import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class GMP:
    def __init__(self, num_user, num_item, K):
        # User embedding
        user_input = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(num_user, K, input_length=user_input.shape[1])(user_input)
        user_embedding = Flatten()(user_embedding)

        # Item embedding
        item_input = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(num_item, K, input_length=item_input.shape[1])(item_input)
        item_embedding = Flatten()(item_embedding)

        # Merge
        GMF_layer = Multiply()([user_embedding, item_embedding])

        # Output
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(GMF_layer)

        # Model
        self.model = Model([user_input, item_input], output_layer)

    def call(self):
        return self.model

    def model_compile(self, optimizer='adam', loss='binary_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)
        return self.model