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

    vad_save_dir = os.path.join(dir_dir_dir_, vad_dir)
    if not os.path.exists(vad_save_dir):
        os.makedirs(vad_save_dir)

    wav_list = glob(wav_recording_dir + '/*.wav')
    print('all wav list length is ', len(wav_list))

    for i in range(len(wav_list)):
        vad_result = vad_tasks(wav_list[i])
        vad_save_path = os.path.join(vad_save_dir, filename.replace(".flac", ".wav"))
        sf.write(vad_save_path, vad_result, SR, format='WAV', endian='LITTLE', subtype='PCM_16')
        print("{} vad saved ".format(vad_save_path))



def text_pre(glob_list):
    for i in range(len(glob_list)):
        _, filename = os.path.split(glob_list[i])
        print('filename is', filename)
        with open(glob_list[i], 'r') as f:
            lines = f.readlines()
        txt = []
        for line in lines:
            print(line)
            txt.append(line)
        print("txt is ", txt)
        exit()



# exit()


all_train_flac = glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/*/*/*.flac')
all_train_txt = glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/*/*/*.txt')
print(len(all_train_flac))
#text_pre(all_train_txt)
sox_and_vad(all_train_flac)

all_valid_flac = glob('/sdd_temp/junewoo10/LibriSpeech/dev-clean/*/*/*.flac')
all_valid_txt = glob('/sdd_temp/junewoo10/LibriSpeech/dev-clean/*/*/*.txt')
print(len(all_valid_flac))
sox_and_vad(all_valid_flac)

all_test_flac = glob('/sdd_temp/junewoo10/LibriSpeech/test-clean/*/*/*.flac')
all_test_txt = glob('/sdd_temp/junewoo10/LibriSpeech/test-clean/*/*/*.txt')
print(len(all_test_flac))
sox_and_vad(all_test_flac)
