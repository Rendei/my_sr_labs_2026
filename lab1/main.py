# Laboratory work 1. Informative features of speech signals: feature extraction
# Run from the ./lab1 directory: python main.py


# Import of modules
import os
import sys

sys.path.append(os.path.realpath('..'))

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import torchaudio

from common import download_dataset, extract_dataset
from exercises_blank import split_meta_line, preemphasis, framing, power_spectrum
from exercises_blank import compute_fbank_filters, compute_fbanks_features, compute_mfcc, mvn_floating


PATH_TO_META = './metadata/meta.txt'
PATH_TO_DATASETS = '../data/lists/datasets.txt'
SAVE_PATH = '../data'
NFILT = 40
NUM_CEPS = 20
NFFT = 512


def prepare_data():
    # Download and extract the VoxCeleb1 test set (only the first line of the list is needed in labs 1-2)
    with open(PATH_TO_DATASETS, 'r') as f:
        lines = f.readlines()

    download_dataset(lines[:1], user='voxceleb1902', password='nx0bl2v2', save_path=SAVE_PATH)

    if not os.path.isdir(os.path.join(SAVE_PATH, 'voxceleb1_test', 'wav')):
        extract_dataset(save_path=os.path.join(SAVE_PATH, 'voxceleb1_test'),
                        fname=os.path.join(SAVE_PATH, 'vox1_test_wav.zip'))


def load_signal(path_to_wav):
    signal, sample_rate = torchaudio.load(path_to_wav)
    signal = signal.numpy().squeeze(axis=0)

    return signal / np.abs(signal).max(), sample_rate


def compute_feats(signal, fbank):
    emphasized_signal = preemphasis(signal)
    frames = framing(emphasized_signal)
    pow_frames = power_spectrum(frames, NFFT)
    filter_banks_features = compute_fbanks_features(pow_frames, fbank)
    mfcc = compute_mfcc(filter_banks_features, num_ceps=NUM_CEPS)

    return filter_banks_features, mfcc


def plot_gender_hist(male_features, female_features, comp_number, name):
    coeff_male = male_features[:, comp_number]
    coeff_female = female_features[:, comp_number]

    min_coeff = min(coeff_male.min(), coeff_female.min())
    max_coeff = min(coeff_male.max(), coeff_female.max())

    plt.figure()
    plt.hist(coeff_male, int(sqrt(len(coeff_male))), histtype='step', color='green',
             range=(min_coeff, max_coeff), density=1)
    plt.hist(coeff_female, int(sqrt(len(coeff_female))), histtype='step', color='red',
             range=(min_coeff, max_coeff), density=1)
    plt.xlabel('{}, component {}'.format(name, comp_number + 1))
    plt.ylabel('Histogram value')
    plt.title('Normalized histograms (green: male, red: female)')
    plt.grid()


def main():
    prepare_data()

    with open(PATH_TO_META, 'r') as f:
        list_lines = f.readlines()
    speaker_ids, genders, paths = zip(*[split_meta_line(line) for line in list_lines[1:]])

    fbank = compute_fbank_filters(nfilt=NFILT, sample_rate=16000, NFFT=NFFT)

    # Features of the first recording
    signal, sample_rate = load_signal(paths[0])
    signal = signal[:int(3.5 * sample_rate)]
    filter_banks_features, mfcc = compute_feats(signal, fbank)
    print('Fbank features shape: {}'.format(filter_banks_features.shape))
    print('MFCC features shape:  {}'.format(mfcc.shape))

    mfcc_cmvn = mvn_floating(mfcc, 150, 150)
    filter_banks_features_mvn = mvn_floating(filter_banks_features, 150, 150)

    plt.figure(figsize=(15, 5))
    plt.subplot(211)
    plt.imshow(filter_banks_features_mvn.T, origin='lower')
    plt.title('Normalized FBanks')
    plt.subplot(212)
    plt.imshow(mfcc_cmvn.T, origin='lower')
    plt.title('Normalized MFCCs')

    # Features of all recordings split by gender
    feats = {'m': ([], []), 'f': ([], [])}
    for path_to_wav, gender in zip(paths, genders):
        signal, _ = load_signal(path_to_wav)
        filter_banks_features, mfcc = compute_feats(signal, fbank)
        feats[gender][0].append(filter_banks_features)
        feats[gender][1].append(mfcc)

    male_fb, male_mfcc = (np.concatenate(x) for x in feats['m'])
    female_fb, female_mfcc = (np.concatenate(x) for x in feats['f'])
    print('Male / female MFB shapes:  {} / {}'.format(male_fb.shape, female_fb.shape))
    print('Male / female MFCC shapes: {} / {}'.format(male_mfcc.shape, female_mfcc.shape))

    for comp_number in range(3):
        plot_gender_hist(male_fb, female_fb, comp_number, 'MFBs')
        plot_gender_hist(male_mfcc, female_mfcc, comp_number, 'MFCCs')

    plt.show()


if __name__ == '__main__':
    main()
