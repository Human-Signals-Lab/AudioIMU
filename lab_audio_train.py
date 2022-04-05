

import time
import numpy as np
import pickle
import os
import random

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
import _pickle as cPickle
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

from ParticipantLab import ParticipantLab as parti
from models import StatisticsContainer, FineTuneCNN14
from utils import plot_confusion_matrix, plotCNNStatistics

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
random.seed(1)


P = 15
win_size = 10
hop = .5

participants = []

# prepare user data
PATH_data = ''

if os.path.exists(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl'):
    with open(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'rb') as f:
        participants = pickle.load(f)
else:
    start = time.time()
    for j in range (1, P+1):
        pname = str(j).zfill(2)
        p = parti(pname, PATH_data, win_size, hop)
        p.readRawAudioMotionData()
        participants.append(p)
        print('participant',j,'data read...')
    end = time.time()
    print("time for feature extraction: " + str(end - start))

    with open(PATH_data + '/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'wb') as f:
        pickle.dump(participants, f)

# load user data
window_size = 1024
hop_size = 320
batch_size = 64 
model_name = 'FineTuneCNN14'
fmin, fmax = 50, 11000
mel_bins = 64
classes_num = 23
sr = 22050
learning_rate = 1e-4
num_epochs = 200
device = 'cuda'
sub_list = np.arange(15)
global_acc, global_f1 = [], []

for sub in sub_list:
    # load training set
    X_trainA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_train = np.zeros((0, 1))

    for u in [participants[sub]]:
        print("training data other than participant (lopo) : " + u.name)            
        for x in participants:
            if x != u:
                X_trainA = np.vstack((X_trainA, x.rawAdataX_s1[:]))
                X_trainA = np.vstack((X_trainA, x.rawAdataX_s2[:]))
                y_train = np.vstack((y_train, x.rawdataY_s1))
                y_train = np.vstack((y_train, x.rawdataY_s2))
        
    # load val set
    X_testA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_test = np.zeros((0, 1))
        
    for u in [participants[sub]]:
        print("test participant (lopo): " + u.name)
        X_testA = np.vstack((X_testA, u.rawAdataX_s1[:]))
        X_testA = np.vstack((X_testA, u.rawAdataX_s2[:]))
        y_test = np.vstack((y_test, u.rawdataY_s1))
        y_test = np.vstack((y_test, u.rawdataY_s2))
    
    # filter out NULL
    y_test = y_test.flatten()
    X_testA = X_testA[y_test != 23]
    y_test = y_test[y_test != 23]	
    
    y_train = y_train.flatten()
    X_trainA = X_trainA[y_train != 23]	
    y_train = y_train[y_train != 23]
    
    y_test = y_test.astype('int64')
    y_train = y_train.astype('int64')
    
    print('training and val size:', np.shape(X_trainA), np.shape(X_testA), np.shape(y_train))

#%%
    torch.cuda.empty_cache()
    
    Model = eval(model_name)
    
    ## CNN
    model = Model(sample_rate=sr, window_size=window_size, 
                hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                classes_num=classes_num)
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    
    if 'cuda' in str(device):
        model.to(device)
    
    PATH_save_models = '/home/dawei/test_files/to_newserver/audioimu_teacher_models/participant_{}'.format(u.name)
    
    x_train_tensor = torch.from_numpy(np.array(X_trainA)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train)).float()
    x_test_tensor = torch.from_numpy(np.array(X_testA)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).float()
    
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                        batch_size=batch_size,
                        num_workers=8, pin_memory=True, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                            batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle = False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    
    iteration = 0
    ### Training Loop ########
    accuracy_stop, stop_cnt = 0, 0
    for epoch in range(num_epochs):
        if stop_cnt == 10:
            break
        for i, d in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = d
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.int64)
    
            model.train()       
            outputs = model(inputs)
            clipwise_output = outputs['clipwise_output']
            loss = F.cross_entropy(clipwise_output, labels)
            loss.backward()
            optimizer.step()
    
        print('[Epoch %d]' % (epoch))
        print('Train loss: {}'.format(loss))
        eval_output = []
        true_output = []
        test_output = []
        true_test_output = []
    
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = torch.from_numpy(np.array(x_val)).float()
                x_val = x_val.to(device)
                y_val = y_val.to(device, dtype=torch.int64)
    
                model.eval()
    
                yhat = model(x_val)
                test_loss = F.cross_entropy(yhat['clipwise_output'], y_val)
    
                test_output.append(yhat['clipwise_output'].data.cpu().numpy())
                true_test_output.extend(y_val.data.cpu().numpy())

            test_oo = np.argmax(np.vstack(test_output), axis = 1)
            true_test_oo = np.asarray(true_test_output)
    
            accuracy = metrics.balanced_accuracy_score(true_test_oo, test_oo)
            # early stopping
            if accuracy_stop < accuracy:
                model_best = copy.deepcopy(model)
                stop_cnt = 0
                accuracy_stop = accuracy
                iteration = epoch
                precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
                try:
                    auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="macro")
                except ValueError:
                    auc_test = None
                print('Test loss: {}'.format(test_loss))
                print('TEST average_precision: {}'.format(precision))
                print('TEST average f1: {}'.format(fscore))
                print('TEST average recall: {}'.format(recall))
                print('TEST acc: {}'.format(accuracy))
        
                trainLoss = {'Trainloss': loss}
                testLoss = {'Testloss': test_loss}
                test_f1 = {'test_f1':fscore}
                        
            else:
                stop_cnt += 1
                
    print('Finished Training')
    global_acc.append(accuracy_stop)
    global_f1.append(fscore)
    
    ### Save model ########
    if not os.path.exists(PATH_save_models):
        os.makedirs(PATH_save_models)
    PATH_save_models = PATH_save_models + '/%s_acc=%.4f_f1=%.4f_epoch%d.pth' % (model_name, accuracy_stop, fscore, iteration)
    torch.save(model_best.state_dict(), PATH_save_models)

print('avg acc, f1:', np.mean(global_acc), np.mean(global_f1))


