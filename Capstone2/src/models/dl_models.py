import numpy as np
import pandas as pd
import pickle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation, Permute, GRU, Reshape
from keras.layers import MaxPooling2D, Flatten, Conv2D, BatchNormalization, Lambda, Bidirectional, concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import keras.utils
from tensorflow import set_random_seed

K.set_image_data_format('channels_last')
set_random_seed(77)
np.random.seed(17)

spec_path = '../input/spect-scaled/all_spects_scaled.npz'
with open(spec_path, 'rb') as handle:
    all_spects = np.load(handle)['arr_0']

info_path = '../input/track-info/track_info.pickle'
with open(info_path, 'rb') as handle:
    track_info = pickle.load(handle)

track_ids = track_info.index
spect_track_id_dict = dict(zip(track_ids, all_spects))

enc = LabelEncoder()
track_info['labels'] = enc.fit_transform(track_info['genre_top'])
y_cat = keras.utils.to_categorical(track_info['labels'].values)
track_info['labels'] = [y.tolist() for y in y_cat]


def make_dataset_splits(split):
    """Return spectrogram arrays for the given dataset split"""
    mask = track_info.split == split
    ids = track_info.loc[mask].index
    data = np.asarray([spect_track_id_dict[i] for i in ids])
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data


def get_input_data(split):

    assert (split == 'training') or (split == 'validation') or (split == 'test') or (split == 'all')

    if split == 'training':
        x_train = make_dataset_splits('training')
        return x_train
    elif split == 'validation':
        x_val = make_dataset_splits('validation')
        return x_val
    elif split == 'test':
        x_test = make_dataset_splits('test')
        return x_test
    else:
        x_train = make_dataset_splits('training')
        x_val = make_dataset_splits('validation')
        x_test = make_dataset_splits('test')
        return x_train, x_val, x_test


def get_input_shape():

    shape = get_input_data(split='test')[0].shape
    return shape


def make_label_splits(split):
    """Return response labels for the given dataset split"""
    mask = track_info.split == split
    labels = track_info.loc[mask, 'labels']
    labels = np.asarray([np.array(l) for l in labels])

    return labels


def get_labels(split):
    assert (split == 'training') or (split == 'validation') or (split == 'test') or (split == 'all')

    if split == 'training':
        y_train = make_label_splits('training')
        return y_train
    elif split == 'validation':
        y_val = make_label_splits('validation')
        return y_val
    elif split == 'test':
        y_test = make_label_splits('test')
        return y_test
    else:
        y_train = make_label_splits('training')
        y_val = make_label_splits('validation')
        y_test = make_label_splits('test')
        return y_train, y_val, y_test


def get_output_shape():
    shape = get_labels(split='test')[0].shape
    return shape


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


def train_model(model,model_name, epochs=20, batch_size=64):
    """
     - Method for training a given Keras model with the given number of training epochs and batch size.
     - Methods saves the weights of the best model to a predetermined directory.
     - Method includes ModelCheckpoint, ReduceLROnPlateau and TensorBoard callbacks.
     - Method returns fitted model and model history"""

    x_train = get_input_data('training')
    x_val = get_input_data('validation')

    y_train = get_labels('training')
    y_val = get_labels('validation')

    # Edit path to desired desitation
    best_model_path = './models/' + model_name + '/weights.best.h5'
    best_model = ModelCheckpoint(best_model_path, monitor='val_acc',
                                 save_best_only=True, mode='max')
    rp_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_delta=0.0)

    tensorboard_path = './models/' + model_name + '/logs'
    tensorboard = TensorBoard(log_dir=tensorboard_path)

    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                     callbacks=[best_model, rp_callback, tensorboard])
    return model, hist


def train_cnn():
    """Train CNN model with validation set preset parameters. Returns the fitted model and training history"""
    param_grid = {'n_filters': 60, 'filter_size': (5, 5), 'pool_size': (2, 2),
                  'input_shape': get_input_shape(),
                  'dropout_rate': 0.3, 'n_cov_layers': 5, 'n_dense_nodes': 240}

    cnn1 = cnn(param_grid)
    cnn_fitted, cnn_history = train_model(cnn1, model_name='cnn1')

    return cnn_fitted, cnn_history


def train_crnn_parallel():
    """Train CRNN model with validation set preset parameters. Returns the fitted model and training history"""
    crnn_parallel_param_grid = dict(n_filters=[16, 32, 64, 64, 64], filter_size=(3, 1),
                                    pool_size=[(2, 2), (2, 2), (2, 2), (4, 4), (4, 4), (4, 2)], dropout_rate=0.3,
                                    n_cov_layers=5, n_recurrent_layers=2, n_time_layers=1, n_time_dst_nodes=128,
                                    n_dense_nodes=240, l2=.001, input_shape=get_input_shape(),
                                    output_shape=get_output_shape())
    crnn_parallel1 = crnn_parallel(crnn_parallel_param_grid)
    crnn_parallel_fitted , crnn_parallel_history = train_model(crnn_parallel1, model_name = 'crnn_parallel1', epochs = 50)

    return crnn_parallel_fitted, crnn_parallel_history


def predict_model(model, model_name):
    """
    Predict the training, validation, and testing accurcary for the given keras model.
    Returns pandas dataframes of the predictions for training, validation, and testing datasets
    Returns pandas series of the macro f1 scores for each dataset split """

    x_train, x_val, x_test = get_input_data('all')
    y_train, y_val, y_test = get_labels('all')

    train_loss, train_acc = model.evaluate(x_train, y_train)
    val_loss, val_acc = model.evaluate(x_val, y_val)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Training set Accuracy: ', train_acc)
    print('Validation set Accuracy: ', val_acc)
    print('Test set Accuracy: ', test_acc)

    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)

    train_pred_classes = train_predictions.argmax(axis=-1)
    val_pred_classes = val_predictions.argmax(axis=-1)
    test_pred_classes = test_predictions.argmax(axis=-1)

    train_f1 = f1_score(y_train, train_pred_classes, average = 'macro')
    val_f1 = f1_score(y_val, val_pred_classes, average='macro')
    test_f1 = f1_score(y_test, test_pred_classes, average='macro')

    f1_scores = pd.Series({'Training': train_f1, 'Validation': val_f1, 'Testing': test_f1})
    f1_scores.name = model_name

    train_predictions_df = pd.DataFrame(train_predictions, columns=[l + '_prob' for l in enc.classes_],
                                        index=track_info.loc[track_info.split == 'training'].index)
    val_predictions_df = pd.DataFrame(val_predictions, columns=[l + '_prob' for l in enc.classes_],
                                      index=track_info.loc[track_info.split == 'validation'].index)
    test_predictions_df = pd.DataFrame(test_predictions, columns=[l + '_prob' for l in enc.classes_],
                                       index=track_info.loc[track_info.split == 'test'].index)

    return train_predictions_df, val_predictions_df, test_predictions_df, f1_scores




