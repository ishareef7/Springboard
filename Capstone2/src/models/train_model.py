import os
import sys

project_path = '/Users/ishareef7/Springboard/Capstone2'
sys.path.append(mod_path)
from src.models import dl_models

spec_path = '../input/spect-scaled/all_spects_scaled.npz'
with open(spec_path, 'rb') as handle:
    all_spects = np.load(handle)['arr_0']

info_path = '../input/track-info/track_info.pickle'
with open(info_path, 'rb') as handle:
    track_info = pickle.load(handle)

def make_dataset_splits(split):
    mask = track_info.split == split
    ids = track_info.loc[mask].index
    data = np.asarray([track_id_dict[i] for i in ids])

    return data

def make_label_splits(split):
    mask = track_info.split == split
    labels = track_info.loc[mask, 'labels']
    labels = np.asarray([np.array(l) for l in labels])
    return labels


def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=20, batch_size=64):
    best_model_path = './models/' + model_name + '/weights.best.h5'
    best_model = ModelCheckpoint(best_model_path, monitor='val_acc',
                                 save_best_only=True, mode='max')
    rp_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_delta=0.0)

    tensorboard_path = './models/' + model_name + '/logs'
    tensorboard = TensorBoard(log_dir=tensorboard_path)

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                     callbacks=[best_model, rp_callback, tensorboard])
    return model, hist
