
import math
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_attention import SelfAttention

"""

The models and processing modules used in paper: 
    AudioIMU: Enhancing Inertial Sensing-Based Activity Recognition with Acoustic Models. ACM ISWC 2022.

Please note that the acoustic encoders and corresponding model implementation in our work are adopted from paper:
    Panns: Large-scale pretrained audio neural networks for audio pattern recognition. By Kong, Qiuqiang, et al.
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.
Please refer to their work for more details, or cite their paper if you use the same acoustic models in your publications.

Thank you!

"""

# ======================= Feature processing modules =======================
class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W
        

class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)

        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag
    
    
class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)

        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max() - self.top_db, max=np.inf)

        return log_spec


# ===================== layer initialization =====================
def init_layer(layer):

    if type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
 
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size, stride=(2,2))
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


# ===================== acoustic teacher =====================
class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print("spectrogram size: {}".format(x.size()))
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        conv_out = F.dropout(x, p=0.2, training=self.training)
        #print('conv output (before all global pooling) from the audio Cnn14 model:', conv_out.shape)
        x = torch.mean(conv_out, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        #print('conv output (after global pooling) from the audio Cnn14 model:', x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding, 'conv_out': conv_out}

        return output_dict


class FineTuneCNN14(nn.Module):
    '''This model is based on Cnn14 architecture above'''
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCNN14, self).__init__()
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)   # Cnn14 architecture
        # load pre-trained weights
        pretrained_checkpoint_path = './Cnn14_mAP=0.431.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])
        self.base.fc_audioset = nn.Linear(2048, classes_num, bias=True)   # final fc layer, no activation required by torch

        self.init_weights()

    def init_weights(self):
        init_layer(self.base.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['embedding']   # pre-trained acoustic embedding
        logits = self.base.fc_audioset(embedding)
        clipwise_output = logits   # clipwise_output is the raw logit output so that softmax can be applied
        
        output_dict = {'clipwise_output': clipwise_output, 'logits':logits, 'embedding': embedding}

        return output_dict


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


# ====================================== student architecture ======================================
class DeepConvLSTM_Split(nn.Module):
    def __init__(self, classes_num, acc_features, gyr_features):
        super(DeepConvLSTM_Split, self).__init__()

        self.name = 'DeepConvLSTM'
        self.convAcc1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc1 = nn.BatchNorm2d(64)
                              
        self.convAcc2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnAcc2 = nn.BatchNorm2d(64)

        self.convAcc3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc3 = nn.BatchNorm2d(64)
                              
        self.convAcc4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnAcc4 = nn.BatchNorm2d(64)

        self.lstmAcc1 = nn.LSTM(64*acc_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmAcc2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)


        self.convGyr1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnGyr1 = nn.BatchNorm2d(64)
                              
        self.convGyr2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr2 = nn.BatchNorm2d(64)

        self.convGyr3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnGyr3 = nn.BatchNorm2d(64)
                              
        self.convGyr4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr4 = nn.BatchNorm2d(64)

        self.lstmGyr1 = nn.LSTM(64*gyr_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmGyr2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)

        self.softmax = nn.Linear(125696, classes_num)   # window=10

        self.init_weight()

    def init_weight(self):
        init_layer(self.convAcc1)
        init_layer(self.convAcc2)
        init_layer(self.convAcc3)
        init_layer(self.convAcc4)
        init_layer(self.convGyr1)
        init_layer(self.convGyr2)
        init_layer(self.convGyr3)
        init_layer(self.convGyr4)

        init_bn(self.bnAcc1)
        init_bn(self.bnAcc2)
        init_bn(self.bnAcc3)
        init_bn(self.bnAcc4)
        init_bn(self.bnGyr1)
        init_bn(self.bnGyr2)
        init_bn(self.bnGyr3)
        init_bn(self.bnGyr4)
        
        init_layer(self.softmax)

    def forward(self, inputAcc, inputGyr):

        x1 = self.convAcc1(inputAcc)
        x1 = self.bnAcc1(x1)
        x1 = self.convAcc2(x1)
        x1 = self.bnAcc2(x1)
        x1 = self.convAcc3(x1)
        x1 = self.bnAcc3(x1)
        x1 = self.convAcc4(x1)
        x1 = self.bnAcc4(x1)
        #print(x1.shape)
        x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
        #print(x1.shape)
        self.lstmAcc1.flatten_parameters()
        x1, _ = self.lstmAcc1(x1)
        self.lstmAcc2.flatten_parameters()
        x1, (h1,c1) = self.lstmAcc2(x1)
        x1 = x1.permute((0,2,1))
        x1 = torch.flatten(x1, start_dim=1)

        x2 = self.convGyr1(inputGyr)
        x2 = self.bnGyr1(x2)
        x2 = self.convGyr2(x2)
        x2 = self.bnGyr2(x2)
        x2 = self.convGyr3(x2)
        x2 = self.bnGyr3(x2)
        x2 = self.convGyr4(x2)
        x2 = self.bnGyr4(x2)      
        x2 =  x2.reshape((x2.shape[0],x2.shape[2],-1)) 
        self.lstmGyr1.flatten_parameters()
        x2, _ = self.lstmGyr1(x2)
        self.lstmGyr2.flatten_parameters()
        x2 , (h2,c2) = self.lstmGyr2(x2)
        #print('lstm output from the motion DeepConvLSTM_Split model:', x2.shape)

        x2 = x2.permute((0,2,1))
        #x2 = self.maxPool2(x2)
        x2 = torch.flatten(x2, start_dim=1)
        #print(x2.shape)
        ##print(".......",x2.shape, x1.shape)
        x3 = torch.cat([x1, x2], 1)
        #print('concat output from the motion DeepConvLSTM_Split model:', x3.shape)
  
        logits = self.softmax(x3)   # self.softmax is just a linear layer, not softmax!
        output = logits

        return {'clipwise_output': output, 'logits': logits, 'embedding': x3}


# ==================================== multimodal teacher ====================================
class Cnn14_AudioExtractor(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14_AudioExtractor, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        # self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        # init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # print("spectrogram size: {}".format(x.size()))
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # import pdb; pdb.set_trace()
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        conv_out = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(conv_out, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))

        return x


class Attention(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.weight1 = torch.nn.Parameter(data = torch.Tensor(dim,size), requires_grad=True)
        self.b1 = torch.nn.Parameter(data = torch.Tensor(1, size), requires_grad=True)
        self.W = torch.nn.Parameter(data = torch.Tensor(size,1), requires_grad=True)

        self.init_weights()


    def init_weights(self):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.b1)


    def forward(self, input, modality=False):
        # print(input.shape, self.weight1.shape)
        x = torch.tensordot(input, self.weight1, dims=1)
        # print(x.shape)
        xx = x + self.b1
        # print(x.shape)
        # import pdb; pdb.set_trace()
        if modality:
            xx_n = torch.linalg.norm(xx, dim=0, keepdims=True)
            # import pdb; pdb.set_trace()
            xx = xx / xx_n
        xx = torch.tanh(xx)
        # print(x.shape)
        alpha = torch.tensordot(xx,self.W, dims=1)
        # attention = torch.softmax(alpha, axis=1)
        output = torch.sum(xx * alpha, dim = 1)
        # print(output.shape)
        return output, alpha
    
    
class DeepConvLSTM_SplitV4(nn.Module):
    def __init__(self, classes_num, acc_features, gyr_features):
        super(DeepConvLSTM_SplitV4, self).__init__()

        self.name = 'DeepConvLSTM_SplitV4'
        self.convAcc1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc1 = nn.BatchNorm2d(64)
                              
        self.convAcc2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnAcc2 = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(kernel_size = (3,1))

        self.convAcc3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnAcc3 = nn.BatchNorm2d(64)
                              
        self.convAcc4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
        self.maxPool2 = nn.MaxPool2d(kernel_size = (3,1))
                          
        self.bnAcc4 = nn.BatchNorm2d(64)

        self.lstmAcc1 = nn.LSTM(64*acc_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmAcc2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)


        self.convGyr1 = nn.Conv2d(in_channels=1, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        self.bnGyr1 = nn.BatchNorm2d(64)
                              
        self.convGyr2 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr2 = nn.BatchNorm2d(64)

        self.convGyr3 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5, 1), stride=(1,1),
                              padding=(0,0))
        
        self.bnGyr3 = nn.BatchNorm2d(64)
        self.maxPool3 = nn.MaxPool2d(kernel_size = (3,1))
                      
        self.convGyr4 = nn.Conv2d(in_channels=64, 
                              out_channels=64,
                              kernel_size=(5,1), stride=(1, 1),
                              padding=(0, 0))
                              
        self.bnGyr4 = nn.BatchNorm2d(64)
        self.maxPool4 = nn.MaxPool2d(kernel_size = (3,1))

        self.lstmGyr1 = nn.LSTM(64*gyr_features, hidden_size=128, num_layers=1, batch_first=True)
        self.lstmGyr2 = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.temporalAttention_Gyr = Attention(128,128)
        self.temporalAttention_Acc = Attention(128,128)

        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.convAcc1)
        init_layer(self.convAcc2)
        init_layer(self.convAcc3)
        init_layer(self.convAcc4)
        init_layer(self.convGyr1)
        init_layer(self.convGyr2)
        init_layer(self.convGyr3)
        init_layer(self.convGyr4)
        #init_layer(self.lstmAcc)
        #init_layer(self.lstmGyr)

        init_bn(self.bnAcc1)
        init_bn(self.bnAcc2)
        init_bn(self.bnAcc3)
        init_bn(self.bnAcc4)
        init_bn(self.bnGyr1)
        init_bn(self.bnGyr2)
        init_bn(self.bnGyr3)
        init_bn(self.bnGyr4)
        #init_layer(self.dense)
        # init_layer(self.softmax)

    def forward(self, inputAcc, inputGyr):

        x1 = self.convAcc1(inputAcc)
        x1 = self.bnAcc1(x1)
        x1 = self.convAcc2(x1)
        x1 = self.bnAcc2(x1)
        #x1 = self.maxPool1(x1)
        #import pdb; pdb.set_trace()
        x1 = self.convAcc3(x1)
        x1 = self.bnAcc3(x1)
        x1 = self.convAcc4(x1)
        x1 = self.bnAcc4(x1)
        x1 = self.maxPool2(x1)
        
        #print(x1.shape)
        x1 =  x1.reshape((x1.shape[0], x1.shape[2],-1)) 
        #print(x1.shape)
        self.lstmAcc1.flatten_parameters()
        x1, _ = self.lstmAcc1(x1)
       
        self.lstmAcc2.flatten_parameters()
        x1, (h1,c1) = self.lstmAcc2(x1)
        x1, alpha1 = self.temporalAttention_Acc(x1)       

        x2 = self.convGyr1(inputGyr)
        x2 = self.bnGyr1(x2)
        x2 = self.convGyr2(x2)
        x2 = self.bnGyr2(x2)
        #x2 = self.maxPool3(x2)

        x2 = self.convGyr3(x2)
        x2 = self.bnGyr3(x2)
        x2 = self.convGyr4(x2)
        x2 = self.bnGyr4(x2) 
        x2 = self.maxPool4(x2)

        x2 =  x2.reshape((x2.shape[0],x2.shape[2],-1)) 
        self.lstmGyr1.flatten_parameters()
        x2, _ = self.lstmGyr1(x2)
        self.lstmGyr2.flatten_parameters()
        x2 , (h2,c2) = self.lstmGyr2(x2)
        x2, alpha2 = self.temporalAttention_Gyr(x2)        

        #print(h2.shape)

        x3 = torch.cat([x1, x2], 1)


        return x3
    
    
class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class DeepConvLSTM_MotionAudio_CNN14_Attention(nn.Module):
    def __init__(
        self,
        model,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax
    ):
        super(DeepConvLSTM_MotionAudio_CNN14_Attention, self).__init__()

        self.fe = DeepConvLSTM_SplitV4(classes_num=num_class, acc_features=3, gyr_features = 3)

        self.audio_fe = Cnn14_AudioExtractor(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, 527)
        self.maxPool = nn.MaxPool1d(kernel_size = 2, stride=2)
        self.sa = SelfAttention(256+2048, sa_div)

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(256+2048, num_class)
        self.register_buffer(
            "centers", (torch.randn(num_class, 256+2048).cuda())
        )

    def forward(self, x_motion, x_audio):
        feature = self.fe(x_motion[:,None,:,:3], x_motion[:,None,:,3:])
        feature_audio = self.audio_fe(x_audio)

        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        z_a = feature_audio.div(
            torch.norm(feature_audio, p=2, dim=1, keepdim=True).expand_as(feature_audio)
        )

        feature = torch.cat([feature,feature_audio], dim=-1)
        z = torch.cat([z, z_a], dim=-1)
        refined = self.sa(feature)

        out = self.dropout(refined)
        logits = self.classifier(out)
        return z, logits
