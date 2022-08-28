

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

from ParticipantLab import ParticipantLab as parti
from models import DeepConvLSTM_MotionAudio_CNN14_Attention
from utils import AverageMeter
from utils_centerloss import compute_center_loss, get_center_delta

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    with open(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'wb') as f:
        pickle.dump(participants, f)

# load user data
window_size = 1024
hop_size = 320
batch_size = 16 #32 #64 
model_name = 'DeepConvLSTM_MotionAudio_CNN14_Attention'   #'Conv_split_individual_MotionAudio_CNN14_Attention'
fmin, fmax = 50, 11000
mel_bins = 64
classes_num = 23
sr = 22050
learning_rate = 1e-4
num_epochs = 100
device = 'cuda'
sub_list = np.arange(15)
global_acc, global_f1 = [], []

config_model = {
    "model": model_name,
    "input_dim": 6,
    "hidden_dim": 128,
    "filter_num": 64,
    "filter_size": 5,
    "enc_num_layers": 2,
    "enc_is_bidirectional": False,
    "dropout": .5,
    "dropout_rnn": .25,
    "dropout_cls": .5,
    "activation": "ReLU",
    "sa_div": 1,
    "num_class": classes_num,
    "train_mode": True,
    "window_size": window_size,
    "hop_size": hop_size,
    "fmin": fmin, 
    "fmax": fmax,
    "mel_bins": mel_bins,
    "sample_rate": sr
}

for sub in sub_list:
    # load training set
    X_trainM = np.empty((0,np.shape(participants[0].rawMdataX_s1)[1], np.shape(participants[0].rawMdataX_s1)[-1]))
    X_trainA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_train = np.zeros((0, 1))

    for u in [participants[sub]]:
        print("training data other than participant (lopo) : " + u.name)            
        for x in participants:
            if x != u:
                X_trainM = np.vstack((X_trainM, x.rawMdataX_s1[:]))
                X_trainM = np.vstack((X_trainM, x.rawMdataX_s2[:]))
                X_trainA = np.vstack((X_trainA, x.rawAdataX_s1[:]))
                X_trainA = np.vstack((X_trainA, x.rawAdataX_s2[:]))
                y_train = np.vstack((y_train, x.rawdataY_s1))
                y_train = np.vstack((y_train, x.rawdataY_s2))
        
    # load val set
    X_testM = np.empty((0,np.shape(participants[0].rawMdataX_s1)[1], np.shape(participants[0].rawMdataX_s1)[-1]))
    X_testA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_test = np.zeros((0, 1))
        
    for u in [participants[sub]]:
        print("test participant (lopo): " + u.name)
        print(X_testM.shape)
        X_testM = np.vstack((X_testM, u.rawMdataX_s1[:]))
        X_testM = np.vstack((X_testM, u.rawMdataX_s2[:]))
        X_testA = np.vstack((X_testA, u.rawAdataX_s1[:]))
        X_testA = np.vstack((X_testA, u.rawAdataX_s2[:]))
        y_test = np.vstack((y_test, u.rawdataY_s1))
        y_test = np.vstack((y_test, u.rawdataY_s2))
        print(X_testM.shape, X_testA.shape, y_test.shape)
    
    # filter out NULL
    y_test = y_test.flatten()
    X_testM = X_testM[y_test != 23]
    X_testA = X_testA[y_test != 23]
    y_test = y_test[y_test != 23]	
    
    y_train = y_train.flatten()
    X_trainM = X_trainM[y_train != 23]
    X_trainA = X_trainA[y_train != 23]	
    y_train = y_train[y_train != 23]
    
    y_test = y_test.astype('int64')
    y_train = y_train.astype('int64')
    
    print('training and val size:', [np.shape(X_trainA), np.shape(X_trainM)], 
                                     [np.shape(X_testA), np.shape(X_testM)], 
                                     np.shape(y_train))

#%%
    torch.cuda.empty_cache()
    
    Model = eval(model_name)
    
    # CNN
    #model = Model(sample_rate=sr, window_size=window_size, 
    #            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
    #            classes_num=classes_num)
    model = Model(**config_model)
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    #model = torch.nn.DataParallel(model)
    
    if 'cuda' in str(device):
        model.to(device)
    
    PATH_save_models = '/home/dawei/test_files/to_newserver/audioimu_teacher_models/multimodal_teacher_DeepConvLSTM-att_cnn14/participant_{}'.format(u.name)
    
    x_trainM_tensor = torch.from_numpy(np.array(X_trainM)).float()
    x_trainA_tensor = torch.from_numpy(np.array(X_trainA)).float()
    y_train_tensor = torch.from_numpy(np.array(y_train)).float()
    x_testM_tensor = torch.from_numpy(np.array(X_testM)).float()
    x_testA_tensor = torch.from_numpy(np.array(X_testA)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).float()
    
    train_data = TensorDataset(x_trainM_tensor, x_trainA_tensor, y_train_tensor)
    test_data = TensorDataset(x_testM_tensor, x_testA_tensor, y_test_tensor)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                        batch_size=batch_size,
                        num_workers=8, pin_memory=True, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                            batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle = False)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    #scheduler = optim.lr_scheduler.StepLR(
     #       optimizer, step_size=10, gamma=0.9)

    iteration = 0
    ### Training Loop ########
    accuracy_stop, stop_cnt = 0, 0
    for epoch in range(num_epochs):
        if stop_cnt == 10:
            break
        #running_loss = 0.0
        if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
            losses = AverageMeter("Loss")
        for i, d in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, inputs_a, labels = d

            if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                centers = model.centers
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            inputs = inputs.to(device)
            inputs_a = inputs_a.to(device)
            labels = labels.to(device, dtype=torch.int64)
    
            model.train()
            if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                z, output_logits = model(inputs, inputs_a)
            else:
                output_logits = model(inputs, inputs_a)
            loss = F.cross_entropy(output_logits, labels)

            if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                # the center loss is introduced to enhance the teacher performance
                center_loss = compute_center_loss(z, centers, labels)
                loss = loss + 0.003 * center_loss
                losses.update(loss.item(), inputs.shape[0])

            loss.backward()
            optimizer.step()

            if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                center_deltas = get_center_delta(z.data, centers, labels, 1e-4)
                model.centers = centers - center_deltas
        #scheduler.step()
    
        print('[Epoch %d]' % (epoch))
        print('Train loss: {}'.format(loss))
        eval_output = []
        true_output = []
        test_output = []
        true_test_output = []
    
        with torch.no_grad():
            for x_val, x_val_a, y_val in test_loader:
                x_val = torch.from_numpy(np.array(x_val)).float()
                x_val_a = torch.from_numpy(np.array(x_val_a)).float()
                x_val = x_val.to(device)
                x_val_a = x_val_a.to(device)
                #val_input = torch.from_numpy(np.array(val_input)).float()
                #val_input = val_input.to(device)
                y_val = y_val.to(device, dtype=torch.int64)
    
                model.eval()
    
                if model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                    z, yhat = model(x_val, x_val_a)
                else:
                    yhat = model(x_val, x_val_a)

                test_loss = F.cross_entropy(yhat, y_val)
    
                test_output.append(yhat.data.cpu().numpy())
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


