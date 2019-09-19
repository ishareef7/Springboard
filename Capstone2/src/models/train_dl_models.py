import os
import sys

project_path = '/Users/ishareef7/Springboard/Capstone2'
sys.path.append(mod_path)
from src.models import dl_models


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=20, batch_size=64):
    # Edit path to desired desitation
    best_model_path = '/models/' + model_name + '/weights.best.h5'
    best_model = ModelCheckpoint(best_model_path, monitor='val_acc',
                                 save_best_only=True, mode='max')
    rp_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_delta=0.0)

    tensorboard_path = '/models/' + model_name + '/logs'
    tensorboard = TensorBoard(log_dir=tensorboard_path)

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                     callbacks=[best_model, rp_callback, tensorboard])
    return model, hist


def train_cnn():
    """Train CNN model with validation set preset parameters. Retuens the fitted model and training history"""
    cnn = dl_models.cnn()
    param_grid = {'n_filters': 60, 'filter_size': (5, 5), 'pool_size': (2, 2), 'input_shape': X_train.shape,
                  'dropout_rate': 0.3, 'n_cov_layers': 5, 'n_dense_nodes': 240}
    cnn = cnn(param_grid)
    cnn_fitted, cnn_history = train_model(cnn1, X_train, y_train, X_val, y_val, model_name='cnn1')

    train_loss_cnn, train_acc_cnn = cnn.evaluate(X_train, y_train)
    val_loss_cnn, val_acc_cnn = cnn.evaluate(X_val, y_val)

    print('Training set Accuracy: ', train_acc_cnn)
    print('Validation set Accuracy: ', val_acc_cnn)

    return cnn_fitted, cnn_history


def train_crnn_parallel():
    """Train CRNN model with validation set preset parameters. Retuens the fitted model and training history"""
    crnn_parallel = dl_models.crnn_parallel()
    crnn_parallel_param_grid = dict(n_filters=[16, 32, 64, 64, 64], filter_size=(3, 1),
                                    pool_size=[(2, 2), (2, 2), (2, 2), (4, 4), (4, 4), (4, 2)], dropout_rate=0.3,
                                    n_cov_layers=5, n_recurrent_layers=2, n_time_layers=1, n_time_dst_nodes=128,
                                    n_dense_nodes=240, l2=.001, output_shape=y_train.shape)
    crnn_parallel = crnn_parallel(X_train, crnn_parallel_param_grid)
    crnn_parallel_fitted , crnn_parallel_history = train_model(crnn_parallel, X_train, y_train, X_val, y_val,
                                                           model_name = 'crnn_parallel2',epochs = 50)

    train_loss_crnn_p, train_acc_crnn_p = crnn_parallel_fitted.evaluate(X_train, y_train)
    val_loss_crnn_p, val_acc_crnn_p = crnn_parallel_fitted.evaluate(X_val, y_val)
    test_loss_crnn_p, test_acc_crnn_p = crnn_parallel_fitted.evaluate(X_test, y_test)

    print('Training set Accuracy: ', train_acc_crnn_p)
    print('Validation set Accuracy: ', val_acc_crnn_p)
    print('Test set Accuracy: ', test_acc_crnn_p)

    return crnn_parallel_fitted , crnn_parallel_history




