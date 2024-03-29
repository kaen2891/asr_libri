"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
from warnings import warn
from numpy import ma

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

PAD = 30
#N_FFT = 512
SAMPLE_RATE = 16000
target_dict = dict() #now, val
target_dict_val = dict() # nbio\\\\
target_dict_other_val = dict() # nbio\\\\
n_mels = 80
N_FFT = 512

np.seterr(divide = 'ignore')
np.seterr(divide = 'warn')

win_length=int(0.032*SAMPLE_RATE) #change 0930_1628
overlap=int(0.016*SAMPLE_RATE) #change 0930_1628

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target

def load_targets_val(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict_val[key] = target
            
def load_targets_other_val(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict_other_val[key] = target


def log_norm(spec):
    ''' log mel spectrogram normalization (mean=0) '''
    
    mel_dim = spec.shape[0]
    time = spec.shape[-1]
    log_S = ma.log10(spec)
    log_S = log_S.filled(0)
    
    
    for k in range(mel_dim):
        dim = log_S[k]
        dim_sum = dim.sum()
        average_dim_for_time = dim_sum / time
        normalize_dim = dim - average_dim_for_time
        
        '''
        if dim_sum == 0:
            normalize_dim = np.array(dim.shape[0])
        else:
            normalize_dim = dim / average_dim_for_time
        '''
        spec[k] = normalize_dim
    return spec

def power_norm(spec):
    #print("input spec shape is", np.shape(spec))
    #exit()
    #spec = np.squeeze(spec)
    #mel_dim = len(spec)
    mel_dim = spec.shape[0]
    time = spec.shape[-1]
    
    for k in range(mel_dim):
        dim = spec[k]
        dim_sum = dim.sum()
        average_dim_for_time = dim_sum / time
        normalize_dim = dim / average_dim_for_time
        '''
        if dim_sum == 0:
            normalize_dim = np.array(dim.shape[0])
        else:
            normalize_dim = dim / average_dim_for_time
        '''
        #print("mel_dim {} time {}".format(mel_dim, time))
        #print("normalize_dim shape is", np.shape(normalize_dim))
        spec[k] = normalize_dim
        #print("k {} dim_sum {} average_dim_for_time {} ".format(k, dim_sum, average_dim_for_time))
        #print("normalize_dim {}".format(normalize_dim))
    #spec = np.expand_dims(spec, axis=0)
    #exit()
    return spec

def get_mel_feature(filepath, norm):  #with power norm
    y, fs = librosa.load(filepath, sr=SAMPLE_RATE)
    
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=n_mels, n_fft=win_length, hop_length=overlap)
    
    if norm == 0:
        S = power_norm(S)
    else:
        S = log_norm(S)
    #print("norm is", norm) 
    feat = torch.FloatTensor(S).transpose(0, 1)
    #feat = torch.FloatTensor(S)
    
    return feat
    
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

def get_script_val(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict_val[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result
    
def get_script_other_val(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict_other_val[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308, norm=0):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.norm = norm
    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_mel_feature(self.wav_paths[idx], self.norm)
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)
        
        return feat, script

class BaseDatasetVal(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308, norm=0):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.norm = 0
    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_mel_feature(self.wav_paths[idx], self.norm)
        script = get_script_val(self.script_paths[idx], self.bos_id, self.eos_id)
        
        return feat, script

class BaseDatasetOtherVal(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308, norm=0):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id
        self.norm = 0
    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_mel_feature(self.wav_paths[idx], self.norm)
        script = get_script_other_val(self.script_paths[idx], self.bos_id, self.eos_id)
        
        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    divide_num = 16
    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]
    max_seq_size = max_seq_sample.size(0)
    
    
    ### add
    if max_seq_size % divide_num !=0:
        max_seq_size = ((max_seq_size//divide_num)+1)*divide_num
    
    if max_seq_size < 480:
        max_seq_size = 480
        
    #print("before, max_seq_size is", max_seq_size)
    max_target_size = len(max_target_sample)
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)
    
    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    #print("Seqs size", seqs.size())
    #print("after, seqs", seqs)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    
    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()
