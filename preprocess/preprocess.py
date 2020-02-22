import os
from glob import glob
import soundfile as sf
import numpy as np
import subprocess
from pyvad import vad, trim
import librosa

SR = 16000


def vad_tasks(wav_list):
    y, sr = librosa.load(wav_list, sr=SR)
    edges = trim(y, SR, fs_vad=SR, hop_length=30, vad_mode=3)

    if isinstance(edges, type(None)):
        print("it can't VAD")
        data = y
    else:
        print("{} VAD Finished. . .".format(wav_list))
        data = y[edges[0]:edges[1]]
    return data


def sox_and_vad(glob_list):
    wav = 'wav'
    vad_dir = 'vad'

    for i in range(len(glob_list)):
        dir_, filename = os.path.split(glob_list[i])
        # print(dir_, filename)
        dir_dir_, dir = os.path.split(dir_)
        dir_dir_dir_, dir_dir = os.path.split(dir_dir_)
        print('filename {} dir {} dir_dir {} dir_dir_dir_ {}'.format(filename, dir, dir_dir,
                                                                     dir_dir_dir_))  # .flac, id, id, dir

        wav_recording_dir = os.path.join(dir_dir_dir_, wav)

        if not os.path.exists(wav_recording_dir):
            os.makedirs(wav_recording_dir)
        wav_recording_path = os.path.join(wav_recording_dir, filename.replace(".flac", ".wav"))
        subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(glob_list[i], str(SR), wav_recording_path)], shell=True)

    print("wav_recording_dir is", wav_recording_dir)
    vad_save_dir = os.path.join(dir_dir_dir_, vad_dir)

    if not os.path.exists(vad_save_dir):
        os.makedirs(vad_save_dir)

    wav_list = glob(wav_recording_dir + '/*.wav')
    print('all wav list length is ', len(wav_list))

    for i in range(len(wav_list)):
        vad_result = vad_tasks(wav_list[i])
        _, name = os.path.split(wav_list[i])
        vad_save_path = os.path.join(vad_save_dir, name)
        sf.write(vad_save_path, vad_result, SR, format='WAV', endian='LITTLE', subtype='PCM_16')
        print("{} vad saved ".format(vad_save_path))


def make_label(glob_file, text_save_dir, label_save_dir, label_name):
    dir_, _ = os.path.split(text_save_dir)
    label_path = os.path.join(dir_, label_name)

    with open(label_path, 'a') as f0:
        char_dict = make_dict()
        with open(glob_file, 'r') as f:
            lines = f.readlines()
        txt = []
        for line in lines:
            txt.append(line)
        for i in range(len(txt)):

            characters = txt[i].rstrip('\n')

            blank_del = characters.split(' ')
            filename = blank_del[0]
            print_txt = ''
            del blank_del[0]
            txt_path = os.path.join(text_save_dir, filename)
            label_path = os.path.join(label_save_dir, filename)
            for ii in range(len(blank_del)):
                print_txt += blank_del[ii] + ' '

            with open(txt_path + '.txt', 'w') as f2:
                f2.write(print_txt)

            load_txt = ''
            for i in range(len(print_txt)):
                char = char_dict[print_txt[i]]
                load_txt += str(char) + ' '
            # print(load_txt)
            load_txt = load_txt[:-3]
            with open(label_path + '.label', 'w') as f3:
                f3.write(load_txt)
            f0.write(filename + ',' + load_txt + '\n')
            # print("")
        # exit()


def make_dict():
    now_dir = os.getcwd()
    up_dir, _ = os.path.split(now_dir)
    char_dict = {}
    all_txt = []
    with open(up_dir + '/librilabel', 'r') as f:
        lines = f.readlines()  # .split('\n')

        for line in lines:
            txt = line.strip()
            all_txt.append(txt)
        for i in range(len(all_txt)):
            this = all_txt[i].split("\t")
            # num = this[0]
            try:
                char = this[1]
            except:
                char = ' '
            char_dict[char] = i
    return char_dict


def text_pre(glob_list, set_label):
    txt_dir = 'txt'
    vad_dir = 'vad'
    for i in range(len(glob_list)):
        dir_, filename = os.path.split(glob_list[i])
        dir_dir_, dir = os.path.split(dir_)
        dir_dir_dir_, dir_dir = os.path.split(dir_dir_)
        # print('filename {} dir {} dir_dir {} dir_dir_dir_ {}'.format(filename, dir, dir_dir, dir_dir_dir_))  # .trans.txt, id, id, dir

        text_save_dir = os.path.join(dir_dir_dir_, txt_dir)
        if not os.path.exists(text_save_dir):
            os.makedirs(text_save_dir)
        label_save_dir = os.path.join(dir_dir_dir_, vad_dir)
        make_label(glob_list[i], text_save_dir, label_save_dir, set_label)

    all_wav = label_save_dir + '/*.wav'
    all_label = label_save_dir + '/*.label'
    
    make_csv(all_wav, all_label)


def make_csv(all_wav, all_label):
    wav = sorted(glob(all_wav))
    label = sorted(glob(all_label))
    
    dir_, _ = os.path.split(wav[0])

    csv_path = os.path.join(dir_, 'data_list.csv')

    # print("label length {} wav length {} csv_path {}".format(len(label), len(wav), csv_path))
    with open(csv_path, 'w', newline='') as f:
        if len(wav) == len(label):
            for i in range(len(wav)):
                _, w = os.path.split(wav[i])
                _, l = os.path.split(label[i])                
                save = w + ',' + l+'\n'
                f.write(save)
                print(save)
    print('{} finished'.format(csv_path))


all_train_flac = glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/*/*/*.flac')
all_train_txt = glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/*/*/*.txt')
train_label = 'train_label'

text_pre(all_train_txt, train_label)
# sox_and_vad(all_train_flac)

all_valid_flac = glob('/sdd_temp/junewoo10/LibriSpeech/dev-clean/*/*/*.flac')
all_valid_txt = glob('/sdd_temp/junewoo10/LibriSpeech/dev-clean/*/*/*.txt')

valid_label = 'valid_label'
# sox_and_vad(all_valid_flac)
text_pre(all_valid_txt, valid_label)

all_test_flac = glob('/sdd_temp/junewoo10/LibriSpeech/test-clean/*/*/*.flac')
all_test_txt = glob('/sdd_temp/junewoo10/LibriSpeech/test-clean/*/*/*.txt')
test_label = 'test_label'
# sox_and_vad(all_valid_flac)
text_pre(all_test_txt, test_label)

print('finished........')
'''

'''



