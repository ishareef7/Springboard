import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation, Permute, GRU, Reshape
from keras.layers import MaxPooling2D, Flatten, Conv2D, BatchNormalization, Lambda, Bidirectional, concatenate
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from tensorflow import set_random_seed

K.set_image_data_format('channels_last')
set_random_seed(77)
np.random.seed(17)


def cnn(param_grid):
    """Method for creating a CNN Keras model object with the parameters given by the the param_grid"""
    n_filters = param_grid['n_filters']
    filter_size = param_grid['filter_size']
    pool_size = param_grid['pool_size']
    dropout_rate = param_grid['dropout_rate']
    n_cov_layers = param_grid['n_cov_layers']
    n_dense_nodes = param_grid['n_dense_nodes']
    input_shape = param_grid['input_shape']

    model = Sequential()

    for i in range(n_cov_layers):
        model.add(Conv2D(filters=n_filters, kernel_size=filter_size, activation='relu',
                         input_shape=(input_shape[-3], input_shape[-2], 1), padding='same',
                         data_format='channels_last', name='conv2d_' + str(i)))
        model.add(BatchNormalization(name='batch_norm_' + str(i)))
        model.add(Activation('relu', name='relu_' + str(i)))
        model.add(MaxPooling2D(pool_size=pool_size, data_format="channels_last", name='max_pool2d_' + str(i)))
        model.add(Dropout(dropout_rate, name='dropout_' + str(i)))

    model.add(Flatten(name='flatten'))

    model.add(Dense(n_dense_nodes, activation='relu', name='dense_1'))
    model.add(Dense(n_dense_nodes, activation='relu', name='dense_2'))
    model.add(Dense(8, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def crnn_parallel(param_grid):
    """Method for creating a CRNN Keras model object with the parameters given by the the param_grid"""
    n_filters = param_grid['n_filters']
    filter_size = param_grid['filter_size']
    pool_size = param_grid['pool_size']
    dropout_rate = param_grid['dropout_rate']
    n_cov_layers = param_grid['n_cov_layers']
    input_shape = param_grid['input_shape']

    input_layer = Input(shape=(input_shape[-3], input_shape[-2], input_shape[-1]),
                        name='input')
    layer = input_layer
    for i in range(n_cov_layers):
        layer = Conv2D(filters=n_filters[i], kernel_size=filter_size, padding='valid',
                       strides=1, name='conv2d_' + str(i))(layer)
        layer = BatchNormalization(axis = 3, name = 'batch_norm' + str(i))(layer)
        layer = Activation('relu', name = 'relu' + str(i))(layer)
        layer = MaxPooling2D(pool_size=pool_size[i], name='pool2d' + str(i))(layer)
        layer = Dropout(dropout_rate, name = 'dropout' + str(i))(layer)

    layer = Flatten()(layer)

    r_layer = MaxPooling2D(pool_size[5], name='pool_lstm')(input_layer)
    r_layer = Lambda(lambda x: K.squeeze(x, axis=-1))(r_layer)

    r_layer = Bidirectional(GRU(64), merge_mode='concat')(r_layer)

    concat = concatenate([layer, r_layer], axis=-1, name='concat')

    output_layer = Dense(8, activation='softmax', name='output')(concat)

    opt = RMSprop(lr=0.0005)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model