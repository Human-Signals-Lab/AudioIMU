
import time
import numpy as np
import pickle
import os
import random
import argparse

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
import copy

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

from models import init_weights, FineTuneCNN14, Conv_split_individual_MotionAudio_CNN14_Attention, DeepConvLSTM_Split, DeepConvLSTM_MotionAudio_CNN14_Attention
from ParticipantLab import ParticipantLab as parti
from utils import plot_confusion_matrix, plotCNNStatistics


#os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
random.seed(1)

#%%

parser = argparse.ArgumentParser(description='training hyper-params')
parser.add_argument('--batch_size', default=250, type=int,
                    help='Name of model to train (default: "countception"')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num_epochs', default=50, type=int, 
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--learning_rate', default=0.001, type=float, 
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--Temp', type=int, default=4, 
                    help='Image patch size (default: None => model default)')
parser.add_argument('--shuffle', type=int, default=1, choices=[0,1],
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N',
                    help='Input image center crop percent (for validation only)')
parser.add_argument('--par', default=None, type=int,
                    help='Participant index, starting from 1')
args = parser.parse_args()

P = 15
win_size = 10 
hop = .5
window_size = 1024
hop_size = 320
fmin, fmax = 50, 11000
mel_bins = 64
classes_num = 23
sr = 22050
device = 'cuda'
learning_rate = args.learning_rate 
num_epochs = args.num_epochs
alpha = args.alpha 
Temp = args.Temp
batch_size = args.batch_size 
sfl=args.shuffle
par = args.par

print('learning_rate:', learning_rate, 'num_epochs:', num_epochs, 'alpha:', alpha, 'Temp:', Temp, 'batch_size:', batch_size, 'sfl:', sfl)

# prepare user data
def prepare_data(PATH_data):
    participants = []
    if os.path.exists(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl'):
        with open(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'rb') as f:
            participants = pickle.load(f)
    else:
        start = time.time()
        for j in range (1, P+1):
            pname = str(j).zfill(2)
            p = parti(pname, PATH_data,win_size, hop)
            p.readRawAudioMotionData()
            participants.append(p)
            print('participant',j,'data read...')
        end = time.time()
        print("time for feature extraction: " + str(end - start))

        with open(PATH_data + './rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test_NEW.pkl', 'wb') as f:
            pickle.dump(participants, f)
    return participants


# load teacher set and model
def load_teacher_model(u, PATH_teacher_models, teacher_model_name):

    window_size = 1024
    hop_size = 320
    fmin, fmax = 50, 11000
    mel_bins = 64
    classes_num = 23
    sr = 22050
    device = 'cuda'

    # this config part is required by the model def    
    config_model = {
    "model": teacher_model_name,
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
 
    # load teacher model
    Model = eval(teacher_model_name)
    # the inpt args are different for different teachers
    if teacher_model_name != 'FineTuneCNN14':
        model = Model(**config_model)
    else:
        model = Model(sample_rate=sr, window_size=window_size, 
                hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                classes_num=classes_num)

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    # the trained DeepConvLSTM_MotionAudio_CNN14_Attention models cannot be loaded with data parallelization
    if teacher_model_name != 'DeepConvLSTM_MotionAudio_CNN14_Attention':
        model = torch.nn.DataParallel(model)
    ckpt_path = [os.path.join(PATH_teacher_models, item) for item in os.listdir(PATH_teacher_models) if item.endswith('.pth')][0]
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model.eval()
    if 'cuda' in str(device):
        model.to(device)

    return model


def load_data(u, participants):
    X_trainM = np.empty((0,np.shape(u.rawMdataX_s1)[1], np.shape(u.rawMdataX_s1)[-1]))   # imu
    X_trainA = np.empty((0,np.shape(u.rawAdataX_s1)[-1]))   # audio
    y_train = np.zeros((0, 1))
    
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
    X_testM = np.empty((0,np.shape(u.rawMdataX_s1)[1], np.shape(u.rawMdataX_s1)[-1]))
    X_testA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_test = np.zeros((0, 1))
    
    print("val participant (lopo): " + u.name)
    X_testM = np.vstack((X_testM, u.rawMdataX_s1[:]))
    X_testM = np.vstack((X_testM, u.rawMdataX_s2[:]))
    X_testA = np.vstack((X_testA, u.rawAdataX_s1[:]))
    X_testA = np.vstack((X_testA, u.rawAdataX_s2[:]))
    y_test = np.vstack((y_test, u.rawdataY_s1))
    y_test = np.vstack((y_test, u.rawdataY_s2))


    # filter out NULL
    y_test = y_test.flatten()
    X_testM = X_testM[y_test != 23]	
    X_testA = X_testA[y_test != 23]
    y_test = y_test[y_test != 23]

    y_train = y_train.flatten()
    X_trainM = X_trainM[y_train != 23]
    X_trainA = X_trainA[y_train != 23]
    y_train = y_train[y_train != 23]	

    # get acc and gyro shape
    X_trainAcc = np.expand_dims(X_trainM[:1,:,:3], axis = 1)
    X_trainGyr = np.expand_dims(X_trainM[:1,:,3:], axis = 1)

    #print('acc and gyro shape:', np.shape(X_trainAcc), np.shape(X_trainGyr))
    print('training and val size:', [np.shape(X_trainA), np.shape(X_trainM)], 
                                     [np.shape(X_testA), np.shape(X_testM)], 
                                     np.shape(y_train))
    #classes = np.unique(y_train).astype(int)

    x_trainM_tensor = torch.from_numpy(np.array(X_trainM)).float()
    x_trainA_tensor = torch.from_numpy(np.array(X_trainA)).float()
    x_testM_tensor = torch.from_numpy(np.array(X_testM)).float()
    x_testA_tensor = torch.from_numpy(np.array(X_testA)).float()
    
    y_test_tensor = torch.from_numpy(np.array(y_test)).long()
    y_train_tensor = torch.from_numpy(np.array(y_train)).long()

    train_data = TensorDataset(x_trainM_tensor, x_trainA_tensor, y_train_tensor)
    test_data = TensorDataset(x_testM_tensor, x_testA_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                        batch_size=batch_size,
                        num_workers=8, pin_memory=True, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                            batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle = False)

    return train_loader, test_loader, [X_trainAcc, X_trainGyr]


def stat_metric(args, acc, f1_macro, u, PATH_log):
    lr = args.learning_rate
    epochs = args.num_epochs
    alp = args.alpha 
    temp = args.Temp
    bs = args.batch_size 
    sfll=args.shuffle  
    par = args.par
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(par,lr,epochs,alp,temp, bs,sfll, acc, f1_macro), 
          file=open(PATH_log + 'participant_{}.logs'.format(u.name),'a+'))


def kd_loss(logits, truth, T=8.0, alpha=0.9):
    # print(T)
    # print(y)

    # truth =F.one_hot(truth, num_classes=23)
    p = F.log_softmax(logits/T, dim=1)
    q = F.softmax(truth/T, dim=1)
    # print(logits.size(), truth.size(), p.size(), q.size())
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / logits.shape[0]
    # print(l_kl * alpha)
    return l_kl * alpha

#%%
    
def main():
    PATH_data = ''
    participants = prepare_data(PATH_data)    
    u = participants[par-1]
    print('handling participant:', u.name)
    
    PATH_teacher_models = '/home/dawei/test_files/to_newserver/audioimu_teacher_models/audio_teacher_lopo/participant_{}/audio'.format(u.name)
    teacher_model_name = 'FineTuneCNN14'   # FineTuneCNN14 or DeepConvLSTM_MotionAudio_CNN14_Attention
    student_model_name = 'DeepConvLSTM_Split'
    PATH_save_models = '/home/dawei/test_files/to_newserver/IMU_models/{}_NEW_kd_expr3_audioteacher/participant_{}'.format(student_model_name, u.name)
    PATH_log = '/home/dawei/test_files/to_newserver/IMU_models/{}_NEW_kd_expr3_audioteacher/'.format(student_model_name)
    
    # prepare teacher model
    teacher_model = load_teacher_model(u, PATH_teacher_models, teacher_model_name)
    # load training and test data    
    train_loader, test_loader, [X_trainAcc, X_trainGyr] = load_data(u, participants)
    # prepare student model
    Model = eval(student_model_name)
    model = Model(classes_num=classes_num, acc_features=np.shape(X_trainAcc)[-1], gyr_features = np.shape(X_trainGyr)[-1])
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    model.apply(init_weights)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.9)

    iteration = 0
    criterion = nn.CrossEntropyLoss()
    ### Training Loop ########
    accuracy_best = 0
    stop_cnt = 0
    for epoch in range(num_epochs):
        if stop_cnt == 20:
            break
        
        for i, d in enumerate(train_loader, 0):
            # get the training inputs; data is a list of [inputs, labels], inputs contain both acc and gyro
            inputs, inputs_a, labels = d
            with torch.no_grad():
                inputs, inputs_a = inputs.to(device), inputs_a.to(device)
                if teacher_model_name == 'DeepConvLSTM_MotionAudio_CNN14_Attention':
                    _, teacher_out = teacher_model(inputs, inputs_a)
                elif teacher_model_name == 'FineTuneCNN14':
                    teacher_out = teacher_model(inputs_a)['logits']
                else:
                    teacher_out = teacher_model(inputs, inputs_a)
                teacher_logits = teacher_out

            # zero the parameter gradients
            optimizer.zero_grad()
            # split acc and gyro inputs for as required by the student
            inputAcc, inputGyr =  np.expand_dims(np.array(inputs[:,:,:3].cpu()), axis = 1), np.expand_dims(np.array(inputs[:,:,3:].cpu()), axis = 1)
            inputAcc, inputGyr =  torch.from_numpy(inputAcc), torch.from_numpy(inputGyr)
            inputAcc = inputAcc.to(device)
            inputGyr = inputGyr.to(device)
            #print(np.shape(inputAcc), np.shape(inputGyr))
            labels = labels.to(device)
            
            # training
            model.train()       
            yhat = model(inputAcc, inputGyr)
            
            loss = criterion(yhat['logits'], labels)*alpha+ (1-alpha)*kd_loss(yhat['logits'], teacher_logits,T=Temp, alpha=1)
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('[Epoch %d]' % (epoch))
        print('Train loss: {}'.format(loss))

        # evaluation
        test_output = []
        true_test_output = []
        with torch.no_grad():
            for i_val, d_val in enumerate(test_loader, 0):
                inputs_val, inputs_a_val, y_val = d_val
                # split acc and gyro inputs for as required by the student
                acc_val, gyr_val =  np.expand_dims(np.array(inputs_val[:,:,:3]), axis = 1), np.expand_dims(np.array(inputs_val[:,:,3:]), axis = 1)               
                acc_val = torch.from_numpy(np.array(acc_val)).float()
                acc_val = acc_val.to(device)
                gyr_val = torch.from_numpy(np.array(gyr_val)).float()
                gyr_val = gyr_val.to(device)
                y_val = y_val.to(device)

                model.eval()
                yhat = model(acc_val, gyr_val)
                test_loss  = criterion(yhat['logits'], y_val)

                # dump the outputs for metric calculation
                test_output.append(yhat['clipwise_output'].data.cpu().numpy())
                if len(true_test_output)>0:
                    true_test_output = np.concatenate((true_test_output, y_val.data.cpu().numpy()), axis=0)
                else:
                    true_test_output=y_val.data.cpu().numpy()

            test_oo = np.argmax(np.vstack(test_output), axis = 1)
            true_test_oo = np.vstack(true_test_output)

            #accuracy = metrics.accuracy_score(true_test_oo, test_oo)
            accuracy = metrics.balanced_accuracy_score(true_test_oo, test_oo)
            # record best metrics/model
            if accuracy_best < accuracy:
                model_best = copy.deepcopy(model)
                train_loss = loss
                stop_cnt = 0
                accuracy_best = accuracy
                iteration = epoch
                #precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
                precision, recall, fscore,_ = copy.deepcopy(metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='macro'))
                try:
                    auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="macro")
                except ValueError:
                    auc_test = None
                print('Test loss: {}'.format(test_loss))
                print('TEST average_precision: {}'.format(precision))
                print('TEST average f1: {}'.format(fscore))
                print('TEST average recall: {}'.format(recall))
                print('TEST acc: {}'.format(accuracy))
    
            else:
                stop_cnt += 1

                
    print('Finished Training')

    ### Save model ########
    accuracy_best, fscore = round(accuracy_best, 4), round(fscore, 4)
    if not os.path.exists(PATH_save_models):
        os.makedirs(PATH_save_models)
    torch.save(model_best.state_dict(), 
               PATH_save_models + '/alp%.2f_tmp%d_acc=%.4f_f1=%.4f_epoch%d.pth' % (args.alpha, args.Temp, accuracy_best, fscore, iteration))

    # write to log
    stat_metric(args, accuracy_best, fscore, u, PATH_log)

main()
