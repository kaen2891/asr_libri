import librosa
import numpy as np
import random
from sparse_image_warp_pytorch import sparse_image_warp
import torch

''' # not use
def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    #print("point_to_warp", point_to_warp)
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    #print("dist_to_warp", dist_to_warp)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    #print("src_pts",src_pts)
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    #print("dest_pts", dest_pts)
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)
'''
#14, 27
def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=1, time_mask_num=2):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    #print("0 = {}".format(mel_spectrogram.shape))  
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    #warped_mel_spectrogram = time_warp(mel_spectrogram)
    warped_mel_spectrogram = mel_spectrogram

    # Step 2 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para) # 0~27
        f = int(f)
        res = int(v - f) #80-27 = 53
        if res <= 0:
            res = 10
        f0 = random.randrange(0, res)
        warped_mel_spectrogram[:, f0:f0+f, :] = 0 # 53~80

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para) #80
        t = int(t)
        res = int(tau - t) #400-80 = 320
        if res <= 0:
            res = 10
        t0 = random.randrange(0, res)
        warped_mel_spectrogram[:, :, t0:t0+t] = 0

    return warped_mel_spectrogram
