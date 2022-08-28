import time
import numpy as np
import pickle
import os
import random

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

from models import init_weights, DeepConvLSTM_Split
from ParticipantLab import ParticipantLab as parti

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
random.seed(1)

#%%
P = 15
win_size = 10
hop = .5

participants = []

# prepare user data
PATH_data = '.'

if os.path.exists(PATH_data + '/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl'):
    with open(PATH_data + '/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'rb') as f:
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

window_size = 1024
hop_size = 320
batch_size = 256 
model_name = 'DeepConvLSTM_Split'

fmin, fmax = 50, 11000
mel_bins = 64
classes_num = 23
sr = 22050
learning_rate = 0.001
num_epochs = 100
device = 'cuda'
sub_list = np.arange(15)
global_acc, global_f1 = [], []

for sub in sub_list:
    # load training set (lopo)
    X_train = np.empty((0,np.shape(participants[0].rawMdataX_s1)[1], np.shape(participants[0].rawMdataX_s1)[-1]))
    y_train = np.zeros((0, 1))
    
    for u in [participants[sub]]:
        print("training data other than participant (lopo) : " + u.name)            
        for x in participants:
            if x != u:
                X_train = np.vstack((X_train, x.rawMdataX_s1[:]))
                X_train = np.vstack((X_train, x.rawMdataX_s2[:]))
                y_train = np.vstack((y_train, x.rawdataY_s1))
                y_train = np.vstack((y_train, x.rawdataY_s2))
    # load val set
    X_test = np.empty((0,np.shape(participants[0].rawMdataX_s1)[1], np.shape(participants[0].rawMdataX_s1)[-1]))
    y_test = np.zeros((0, 1))
        
    for u in [participants[sub]]:
        print("test participant (lopo): " + u.name)
        X_test = np.vstack((X_test, u.rawMdataX_s1[:]))
        X_test = np.vstack((X_test, u.rawMdataX_s2[:]))
        y_test = np.vstack((y_test, u.rawdataY_s1))
        y_test = np.vstack((y_test, u.rawdataY_s2))
    
    # filter out NULL
    y_test = y_test.flatten()
    X_test = X_test[y_test != 23]	
    y_test = y_test[y_test != 23]
    
    y_train = y_train.flatten()
    X_train = X_train[y_train != 23]
    y_train = y_train[y_train != 23]	
    
    # split acc and gyro
    X_trainAcc = np.expand_dims(X_train[:,:,:3], axis = 1)
    X_trainGyr = np.expand_dims(X_train[:,:,3:], axis = 1)
    X_testAcc = np.expand_dims(X_test[:,:,:3], axis = 1)
    X_testGyr = np.expand_dims(X_test[:,:,3:], axis = 1)
    
    y_test = y_test.astype('int64')
    y_train = y_train.astype('int64')
    
    print('acc and gyro shape:', np.shape(X_trainAcc), np.shape(X_trainGyr))
    print('training and val size:', np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))


#%%
    torch.cuda.empty_cache()
    
    Model = eval(model_name)
    
    ## CNN
    para = 0
    model = Model(classes_num=classes_num, acc_features=np.shape(X_trainAcc)[-1], gyr_features = np.shape(X_trainGyr)[-1])
    para += sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_name, 'model size:', para)
    #model = Model(n_classes = classes_num)
    # Parallel
    # print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)
    
    if 'cuda' in str(device):
        model.to(device)
    model.apply(init_weights)
    
    PATH_save_models = '/home/dawei/test_files/to_newserver/IMU_models/{}_NEW2/participant_{}'.format(model_name, u.name)
    PATH_log = '/home/dawei/test_files/to_newserver/IMU_models/{}_NEW2/results.logs'.format(model_name)
    
    x_trainAcc_tensor = torch.from_numpy(np.array(X_trainAcc)).float()
    x_trainGyr_tensor = torch.from_numpy(np.array(X_trainGyr)).float()
    #x_train_tensor = torch.from_numpy(np.array(X_trainA)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train)).float()
    x_testAcc_tensor = torch.from_numpy(np.array(X_testAcc)).float()
    x_testGyr_tensor = torch.from_numpy(np.array(X_testGyr)).float()
    #x_test_tensor = torch.from_numpy(np.array(X_testA)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).float()
    
    train_data = TensorDataset(x_trainAcc_tensor, x_trainGyr_tensor, y_train_tensor)
    test_data = TensorDataset(x_testAcc_tensor, x_testGyr_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                        batch_size=batch_size,
                        num_workers=1, pin_memory=True, shuffle = False)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                            batch_size=batch_size,
                            num_workers=1, pin_memory=True, shuffle = False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)

    iteration = 0
    ### Training Loop ########
    accuracy_stop, stop_cnt = 0, 0
    for epoch in range(num_epochs):
        if stop_cnt == 20:
            break
        for i, d in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputAcc, inputGyr, labels = d
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            inputAcc = inputAcc.to(device)
            inputGyr = inputGyr.to(device)
            #print(np.shape(inputAcc), np.shape(inputGyr))
            labels = labels.to(device, dtype=torch.int64)
            model.train()       
    
            #print(np.shape(labels))
            yhat = model(inputAcc, inputGyr)
            loss = F.cross_entropy(yhat['clipwise_output'], labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
        print('[Epoch %d]' % (epoch))
        print('Train loss: {}'.format(loss))
        eval_output = []
        true_output = []
        test_output = []
        true_test_output = []
    
        with torch.no_grad():
    
            for acc_val, gyr_val, y_val in test_loader:
    
                acc_val = torch.from_numpy(np.array(acc_val)).float()
                acc_val = acc_val.to(device)
                gyr_val = torch.from_numpy(np.array(gyr_val)).float()
                gyr_val = gyr_val.to(device)
                y_val = y_val.to(device, dtype=torch.int64)
    
                model.eval()
                yhat = model(acc_val, gyr_val)
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
                precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='macro')
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
    print('{}\t{}\t{}'.format(sub, accuracy_stop, fscore), file=open(PATH_log,'a+'))

print('avg acc, f1:', np.mean(global_acc), np.mean(global_f1))
print('{}\t{}\t{}'.format('Average', np.mean(global_acc), np.mean(global_f1)), file=open(PATH_log,'a+'))





