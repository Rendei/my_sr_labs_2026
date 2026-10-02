# Laboratory work 2. Voice activity detector training
# Run from the ./lab2 directory: python main.py


# Import of modules
import os
import sys

sys.path.append(os.path.realpath('..'))

from math import sqrt, pi

import numpy as np
import torchaudio
from torchaudio.transforms import Resample

from common import download_dataset, extract_dataset
from exercises_blank import load_vad_markup, energy_gmm_vad, reverb, awgn


PATH_TO_DATASETS = '../data/lists/datasets.txt'
SAVE_PATH = '../data'
PATH_TO_WAV = '../data/voxceleb1_test/wav'
PATH_TO_RTTM = './ground_truth/rttm'
PATH_TO_IR = './cathIR.wav'

# Parameters of voice activity detector
WINDOW = 320
SHIFT = 160
N_REALIGNMENT = 10
VAD_THR = 0.3
MASK_SIZE_MORPH_FILT = 6000
SIGMA_NOISE = 0.15

# Gaussian probability density function
gauss_pdf = lambda value, m, sigma: 1 / ((abs(sigma) + 1e-10) * sqrt(2 * pi)) * np.exp(
    -(value - m) ** 2 / (2 * (abs(sigma) + 1e-10) ** 2))


def prepare_data():
    # Download and extract the VoxCeleb1 test set (only the first line of the list is needed in labs 1-2)
    with open(PATH_TO_DATASETS, 'r') as f:
        lines = f.readlines()

    download_dataset(lines[:1], user='voxceleb1902', password='nx0bl2v2', save_path=SAVE_PATH)

    if not os.path.isdir(PATH_TO_WAV):
        extract_dataset(save_path=os.path.join(SAVE_PATH, 'voxceleb1_test'),
                        fname=os.path.join(SAVE_PATH, 'vox1_test_wav.zip'))


def load_impulse_response(fs):
    impulse_response, ir_fs = torchaudio.load(PATH_TO_IR)
    impulse_response = Resample(orig_freq=ir_fs, new_freq=fs)(impulse_response)
    impulse_response = impulse_response.numpy().squeeze(axis=0)

    return impulse_response / np.abs(impulse_response).max()


def evaluate(aug_mode=None):
    # Compute VAD quality on all recordings which have ideal markup, aug_mode: None, 'reverb' or 'awgn'
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0

    for rttm_file in os.listdir(PATH_TO_RTTM):
        path_to_wav = os.path.join(PATH_TO_WAV, rttm_file[:7], rttm_file[8:19], '.'.join([rttm_file[20:25], 'wav']))
        signal, fs = torchaudio.load(path_to_wav)
        signal = signal.numpy().squeeze(axis=0)
        signal = signal / np.abs(signal).max()

        if aug_mode == 'reverb':
            signal = reverb(signal, load_impulse_response(fs))
        elif aug_mode == 'awgn':
            signal = awgn(signal, SIGMA_NOISE)

        vad_markup_ideal = load_vad_markup(os.path.join(PATH_TO_RTTM, rttm_file), signal, fs)
        vad_markup_real = energy_gmm_vad(signal, WINDOW, SHIFT, gauss_pdf, N_REALIGNMENT, VAD_THR,
                                         MASK_SIZE_MORPH_FILT)

        TP += np.sum(vad_markup_ideal * vad_markup_real)
        FP += np.sum((1 - vad_markup_ideal) * vad_markup_real)
        FN += np.sum(vad_markup_ideal * (1 - vad_markup_real))
        TN += np.sum((1 - vad_markup_ideal) * (1 - vad_markup_real))

    return FN / (FN + TP), FP / (FP + TN), TP / (TP + FP), TP / (TP + FN)


def main():
    prepare_data()
    np.random.seed(0)

    for aug_mode in (None, 'awgn', 'reverb'):
        FNR, FPR, P, R = evaluate(aug_mode)

        print('Augmentation:         {}'.format(aug_mode or 'none (clean data)'))
        print('Threshold value:      {0:.3f}'.format(VAD_THR))
        print('False negative rate:  {0:.3f}'.format(FNR))
        print('False positive rate:  {0:.3f}'.format(FPR))
        print('Precision:            {0:.3f}'.format(P))
        print('Recall:               {0:.3f}'.format(R))
        print()


if __name__ == '__main__':
    main()
