import time
import queue
import Levenshtein as Lev
import os

import argparse

parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser(desc5iption='Speech hackathon Baseline')
parser.add_argument('--gpu', type=str, default=0)
# parser.add_argument("--pause", type=int, default=0)

args = parser.parse_args()
import random
import spec_augment_pytorch
import sparse_image_warp_pytorch

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length


def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    #save_file = 'result.txt'
    f = open('./result.txt', 'a', newline='')
    
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])  # string
        print("ref = ", ref)
        hyp = label_to_string(hyp_labels[i])  # predict
        print("hyp = ", hyp)
        f.write("real = {}, predict = {} \n".format(ref, hyp))
        
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    f.close()
    return total_dist, total_length


def evaluate(model, dataloader, queue, criterion, device):
    # logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            logit = model(feats, scripts[:, :-1], mode='eval')
            logit = logit[:, :target.size(1), :]

            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    # logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length


import label_loader
import random
from loader import *

import torch
import torch.nn as nn
import torch.optim as optim

import queue

# from models.transformer import Model  # 2d mel style vgg
from models.transformer_3d import Model  # 3d CNN
from models.utils import ScheduledOptim, LabelSmoothingLoss

DATASET_PATH = '/mnt/junewoo/dataset/naver/test/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

char2index, index2char = label_loader.load_label("hackathon.labels")
SOS_token = char2index['<s>']
EOS_token = char2index['</s>']
PAD_token = char2index['_']

char2index['[MASK]'] = len(char2index)
index2char[len(index2char)] = '[MASK]'

MASK_token = char2index['[MASK]']


teacher_forcing = True

##############################################################
model_path = '/mnt/junewoo/workspace/asr_hackathon/MelStyle_model/model_dir/1th_model'
batch_size = 20
# d_model_size = 1536
d_model_size = 1536
model = Model(len(char2index), SOS_token, EOS_token, d_model=d_model_size, nhead=8, max_seq_len=1024,
              num_encoder_layers=0, num_decoder_layers=6,
              enc_feedforward=2048, dec_feedforward=2048,
              dropout=0.1, padding_idx=PAD_token, mask_idx=MASK_token, device=device)
model.load_state_dict(torch.load(model_path))
model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=args.lr)
#okay..
#optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3, lr=args.lr)

# criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
# criterion = nn.NLLLoss(reduction='sum', ignore_index=PAD_token).to(device)
criterion = LabelSmoothingLoss(0.1, len(char2index), ignore_index=PAD_token).to(device)

# **************************************************************************
data_list = os.path.join(DATASET_PATH, 'test_set.csv')

wav_paths = list()
script_paths = list()

with open(data_list, 'r') as f:
    for line in f:
        # line: "aaa.wav,aaa.label"
        wav_path, script_path = line.strip().split(',')
        #wav_paths.append(os.path.join(DATASET_PATH, 'train/train_data', wav_path))
        wav_paths.append(os.path.join(DATASET_PATH, wav_path))
        #script_paths.append(os.path.join(DATASET_PATH, 'train/train_data', script_path))
        script_paths.append(os.path.join(DATASET_PATH, script_path))
#target_path = os.path.join(DATASET_PATH, 'train/train_label')
target_path = os.path.join(DATASET_PATH, 'test_label')

load_targets(target_path)


test_dataset = BaseDataset(wav_paths, script_paths, SOS_token, EOS_token)
test_queue = queue.Queue(3 * 2)
test_loader = BaseDataLoader(test_dataset, test_queue, batch_size, 0)
test_loader.start()

#load_targets(target_path)

# **************************************************************************

#save_dir = '/mnt/junewoo/workspace/asr_hackathon/MelStyle_model/model_dir/{}th_model'.format(args.file_num)
#print("save path is", save_dir)
#train_begin = time.time()
#for epoch in range(epochs):
eval_loss, eval_cer = evaluate(model, test_loader, test_queue, criterion, device)
print('Evaluate CER %0.4f' % (eval_cer))
print('CRR %0.4f' % (1.0-(eval_cer)))
#summary.add_scalar('eval_loss', eval_loss, epoch)
#summary.add_scalar('eval_cer', eval_cer, epoch)
#torch.save(model.state_dict(), save_dir)

#valid_loader.join()
