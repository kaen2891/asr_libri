from glob import glob
import os
label = sorted(glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/vad/*.label'))
wav = sorted(glob('/sdd_temp/junewoo10/LibriSpeech/train-clean-360/vad/*.wav'))

for i in range(10):
    _, l = os.path.split(label[i])
    _, w = os.path.split(wav[i])
    save = w+','+l
    with open('./data_list.csv', 'w', newline='') as f:
        f.write(save)
    print(save)
