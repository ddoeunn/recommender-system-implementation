import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class NMF():
    def __init__(self, num_user, num_item, K, drop_rate=0.2):

        user_input = Input(shape=(1,), dtype='int32')
        item_input = Input(shape=(1,), dtype='int32')

        # User embedding (GMF)
        user_input = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(num_user, K, input_length=user_input.shape[1])(user_input)
        user_embedding = Flatten()(user_embedding)

        # Item embedding (GFM)
        item_input = Input(shape=(1,), dtype='int32')
        item_embedding = Embedding(num_item, K, input_length=item_input.shape[1])(item_input)
        item_embedding = Flatten()(item_embedding)

        # Merge
        GMF_layer = Multiply()([user_embedding, item_embedding])


        # User embedding (MLP)
        user_embedding = Embedding(num_user, 32, input_length=user_input.shape[1])(user_input)
        user_embedding = Flatten()(user_embedding)

        # Item embedding (MLP)
        item_embedding = Embedding(num_item, 32, input_length=item_input.shape[1])(item_input)
        item_embedding = Flatten()(item_embedding)

        # Layer0
        MLP_layer0 = Concatenate(name='layer0')([user_embedding, item_embedding])
        dropout0 = Dropout(rate=drop_rate, name='dropout0')(MLP_layer0)

        # Layer1
        MLP_layer1 = Dense(units=64, activation='relu', name='layer1')(dropout0)  # (64,1)
        dropout1 = Dropout(rate=drop_rate, name='dropout1')(MLP_layer1)
        batch_norm1 = BatchNormalization(name='batch_norm1')(dropout1)

        # Layer2
        MLP_layer2 = Dense(units=32, activation='relu', name='layer2')(batch_norm1)  # (32,1)
        dropout2 = Dropout(rate=drop_rate, name='dropout2')(MLP_layer2)
        batch_norm2 = BatchNormalization(name='batch_norm2')(dropout2)

        # Layer3
        MLP_layer3 = Dense(units=16, activation='relu', name='layer3')(batch_norm2)  # (16,1)

        # Layer4
        MLP_layer3 = Dense(units=8, activation='relu', name='layer4')(MLP_layer3)  # (8,1)

        # merge GMF + MLP
        NMF_layer = tf.keras.layers.concatenate([GMF_layer, MLP_layer3])

        # Output layer
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(NMF_layer)

        # Model
        self.model = Model([user_input, item_input], output_layer)

    def call(self):
        return self.model

    def model_compile(self, optimizer='adam', loss='binary_crossentropy'):
         self.model.compile(optimizer= optimizer, loss= loss)
         return self.model

