import torch
import torch.optim as optim
import torch.utils.data as data_utils
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import csv
import pandas as pd
import matplotlib.pyplot as plt
import glob
import gc
import h5py
import pickle as pk

from utils import log_results, SaveBestModel, train_seq, test_seq
from utils import normalize_mel_sp_slides

from models import cnn_rnn



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataDir = './data'
resultsDir = 'Results'
tempDir = 'temp'

if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)
if not os.path.exists(tempDir):
    os.makedirs(tempDir)

##Loading data from h5 file which we preprocessed and saved earlier
fname = 'birds_xeno_spectr_slide_105_species_sr_32000_len_7_sec_500_250_New.h5'
fileLoc = os.path.join(dataDir,fname)
hf = h5py.File(fileLoc, 'r')
mel_sp = hf.get('mel_spectr')[()]
metadata_total = pd.read_hdf(fileLoc, 'info')
hf.close()  



original_label = list(metadata_total['species'])
lb_bin = LabelBinarizer()
lb_enc = LabelEncoder()
labels_one_hot = lb_bin.fit_transform(original_label)
labels_multi_lbl = lb_enc.fit_transform(original_label)

number_of_sample_classes = len(lb_enc.classes_)
print("Number of Species: ", number_of_sample_classes)
species_id_class_dict_tp = dict()
for (class_label, species_id) in enumerate(lb_bin.classes_):
    species_id_class_dict_tp[species_id] = class_label




mel_sp_normalized = []
for i in range(len(mel_sp)):
    xx_ = normalize_mel_sp_slides(mel_sp[i]).astype('float32')
    mel_sp_normalized += [np.expand_dims(xx_, axis=-3)]
mel_sp_normalized = np.array(mel_sp_normalized)


batch_size = 16*2
shuffleBatches=True
num_epoch = 20


cfg_cnn = [32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'] # CNN1
# n_units = 128*2

cfg_cnn2 = [32, 64, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M']
# n_units = 256*2

##Considering cnn3 config for training
cfg_cnn3 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'] # CNN3
n_units = 512*2 #512*2



##Taking hidden size as 256 and epochs as 20 and selecting only lstm cell as they are much more intelligent as compared to gru and others
hidden_size = 256 #512
rnnConfigs = [
    {'LSTM_0':{'input_size':n_units, 'h_states_ctr':2},
    'LSTM_1':{'input_size':hidden_size, 'h_states_ctr':2}  # 2 layers of LSTM cell
    },
]


##Training Part
exp_no_base = 0
exp_ctr = 0
for ii, cfg in enumerate(rnnConfigs):
    exp_ctr += 1

    exp_no = exp_no_base + exp_ctr
    log_file_name = f'100_species_spectr_cnn_rnn_7sec_h_{hidden_size}_nl_{ii+1}_{exp_no}.p'
    store_ = log_results(file_name=log_file_name, results_dir = resultsDir)
    PATH_curr = os.path.join(tempDir, f'currentModel_cnn_rnn_{exp_no}.pt')
    saveModel = SaveBestModel(PATH=PATH_curr, monitor=-np.inf, verbose=True)

    exp_ind = 0
    skf = StratifiedKFold(n_splits=5, random_state=42)
    for train_ind, test_ind in skf.split(mel_sp_normalized, labels_multi_lbl):

        PATH_curr = os.path.join(tempDir, f'currentModel_cnn_rnn_{exp_no}_{exp_ind}.pt')
        saveModel = SaveBestModel(PATH=PATH_curr, monitor=-np.inf, verbose=True)

        X_train, X_test_p_valid = mel_sp_normalized[train_ind,:], mel_sp_normalized[test_ind,:]

        y_train, y_test_p_valid = labels_one_hot[train_ind], labels_one_hot[test_ind]
        y_train_mlbl, y_test_p_valid_mlbl = labels_multi_lbl[train_ind], labels_multi_lbl[test_ind]
        X_valid, X_test, y_valid, y_test = train_test_split(X_test_p_valid, y_test_p_valid,
                                                               test_size=0.5,
                                                               stratify=y_test_p_valid_mlbl,
                                                               random_state=42)

        print('X_train shape: ', X_train.shape)
        print('X_valid shape: ', X_valid.shape)
        print('X_test shape: ', X_test.shape)

        X_train, X_valid  = torch.from_numpy(X_train).float(), torch.from_numpy(X_valid).float()
        y_train, y_valid = torch.from_numpy(y_train), torch.from_numpy(y_valid)

        y_train, y_valid = y_train.float(), y_valid.float()
        train_use = data_utils.TensorDataset(X_train, y_train)
        train_loader = data_utils.DataLoader(train_use, batch_size=batch_size, shuffle=shuffleBatches)

        val_use = data_utils.TensorDataset(X_valid, y_valid)
        val_loader = data_utils.DataLoader(val_use, batch_size=32, shuffle=False)
        
        print("\n\nModels RNN config\n",cfg,"\n\n")

        model = cnn_rnn(cnnConfig = cfg_cnn3, 
                        rnnConfig = cfg, 
                        hidden_size=hidden_size, 
                        # order=order,
                        # theta=theta,
                        num_classes=105)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=1e-7)

        val_acc_epochs = []
        val_loss_epochs = []
        for epoch in range(1, num_epoch+1):
            train_loss = train_seq(model, train_loader, optimizer, epoch, 
                                    device,
                                    verbose=1, loss_fn = 'bceLogit')
            val_loss, val_acc = test_seq(model, val_loader,
                                        device,
                                        loss_fn = 'bceLogit')
            val_acc_epochs.append(val_acc)
            val_loss_epochs.append(val_loss)
            print("Epoch:",epoch)
            print('val loss = %f, val acc = %f'%(val_loss, val_acc))
            saveModel.check(model, val_acc, comp='max')

        # loading best validated model
        model = cnn_rnn(cnnConfig = cfg_cnn3, 
                        rnnConfig = cfg, 
                        hidden_size=hidden_size, 
                        # order=order,
                        # theta=theta,
                        num_classes=105)
        model.to(device)
        model.load_state_dict(torch.load(PATH_curr))

        X_test, y_test  = torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        test_use = data_utils.TensorDataset(X_test, y_test)
        test_loader = data_utils.DataLoader(test_use, batch_size=32, shuffle=False)
        test_loss, test_acc = test_seq(model, test_loader,
                                    device,
                                    loss_fn = 'bceLogit')
        print('test loss = %f, test acc = %f'%(test_loss, test_acc))

        log_ = dict(
                exp_ind = exp_ind,
                epochs = num_epoch,
                validation_accuracy = val_acc_epochs,
                validation_loss = val_loss_epochs,
                test_loss = test_loss,
                test_accuracy = test_acc,
                X_train_shape = X_train.shape,
                X_valid_shape = X_valid.shape,
                batch_size =batch_size,
        )
        store_.update(log_)
        exp_ind += 1    






