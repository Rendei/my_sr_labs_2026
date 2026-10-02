# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
import scipy.signal

from skimage.morphology import opening, closing


def load_vad_markup(path_to_rttm, signal, fs):
    # Function to read rttm files and generate VAD's markup in samples
    
    vad_markup = np.zeros(len(signal)).astype('float32')
        
    ###########################################################
    # Here is your code
    with open(path_to_rttm, 'r') as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue

            start = int(round(float(fields[3]) * fs))                      # start of speech segment in samples
            stop = int(round((float(fields[3]) + float(fields[4])) * fs))  # end of speech segment in samples
            vad_markup[start:stop] = 1
    ###########################################################

    return vad_markup

def framing(signal, window=320, shift=160):
    # Function to create frames from signal

    shape   = (int((signal.shape[0] - window)/shift + 1), window)
    frames  = np.zeros(shape).astype('float32')

    ###########################################################
    # Here is your code
    frames = np.lib.stride_tricks.sliding_window_view(signal, window)[::shift][:shape[0]].astype('float32')
    ###########################################################

    return frames

def frame_energy(frames):
    # Function to compute frame energies

    E = np.zeros(frames.shape[0]).astype('float32')

    ###########################################################
    # Here is your code
    E = np.sum(frames, axis=1).astype('float32')  # frames are already squared in energy_gmm_vad
    ###########################################################

    return E

def norm_energy(E):
    # Function to normalize energy by mean energy and energy standard deviation

    E_norm = np.zeros(len(E)).astype('float32')

    ###########################################################
    # Here is your code
    E_norm = ((E - np.mean(E)) / np.std(E, ddof=1)).astype('float32')
    ###########################################################

    return E_norm

def gmm_train(E, gauss_pdf, n_realignment):
    # Function to train parameters of gaussian mixture model
    
    # Initialization gaussian mixture models
    w     = np.array([ 0.33, 0.33, 0.33])
    m     = np.array([-1.00, 0.00, 1.00])
    sigma = np.array([ 1.00, 1.00, 1.00])

    g = np.zeros([len(E), len(w)])
    for n in range(n_realignment):

        # E-step
        ###########################################################
        # Here is your code
        for j in range(len(w)):
            g[:, j] = w[j] * gauss_pdf(E, m[j], sigma[j])
        g = g / np.maximum(np.sum(g, axis=1, keepdims=True), np.finfo(float).tiny)
        ###########################################################

        # M-step
        ###########################################################
        # Here is your code
        M = len(E)
        w = np.sum(g, axis=0) / M
        m = np.sum(g * E[:, np.newaxis], axis=0) / (M * w)
        sigma = np.sqrt(np.sum(g * (E[:, np.newaxis] - m) ** 2, axis=0) / (M * w))
        ###########################################################

    return w, m, sigma

def eval_frame_post_prob(E, gauss_pdf, w, m, sigma):
    # Function to estimate a posterior probability that frame isn't speech

    g0 = np.zeros(len(E))

    ###########################################################
    # Here is your code
    likelihoods = np.array([w[j] * gauss_pdf(E, m[j], sigma[j]) for j in range(len(w))])  # [ncomponents x nframes]
    g0 = likelihoods[0] / np.maximum(np.sum(likelihoods, axis=0), np.finfo(float).tiny)
    ###########################################################

    return g0

def energy_gmm_vad(signal, window, shift, gauss_pdf, n_realignment, vad_thr, mask_size_morph_filt):
    # Function to compute markup energy voice activity detector based of gaussian mixtures model
    
    # Squared signal
    squared_signal = signal**2
    
    # Frame signal with overlap
    frames = framing(squared_signal, window=window, shift=shift)
    
    # Sum frames to get energy
    E = frame_energy(frames)
    
    # Normalize the energy
    E_norm = norm_energy(E)
    
    # Train parameters of gaussian mixture models
    w, m, sigma = gmm_train(E_norm, gauss_pdf, n_realignment=n_realignment)
    
    # Estimate a posterior probability that frame isn't speech
    g0 = eval_frame_post_prob(E_norm, gauss_pdf, w, m, sigma)
    
    # Compute real VAD's markup
    vad_frame_markup_real = (g0 < vad_thr).astype('float32')  # frame VAD's markup

    vad_markup_real = np.zeros(len(signal)).astype('float32') # sample VAD's markup
    for idx in range(len(vad_frame_markup_real)):
        vad_markup_real[idx*shift:shift+idx*shift] = vad_frame_markup_real[idx]

    vad_markup_real[len(vad_frame_markup_real)*shift - len(signal):] = vad_frame_markup_real[-1]
    
    # Morphology Filters
    vad_markup_real = closing(vad_markup_real, np.ones(mask_size_morph_filt)) # close filter
    vad_markup_real = opening(vad_markup_real, np.ones(mask_size_morph_filt)) # open filter
    
    return vad_markup_real

def reverb(signal, impulse_response):
    # Function to create reverberation effect
    
    signal_reverb = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    signal_reverb = scipy.signal.fftconvolve(signal, impulse_response)[:len(signal)].astype('float32')
    ###########################################################

    return signal_reverb

def awgn(signal, sigma_noise):
    # Function to add white gaussian noise to signal
    
    signal_noise = np.zeros(len(signal)).astype('float32')
    
    ###########################################################
    # Here is your code
    signal_noise = (signal + sigma_noise * np.random.randn(len(signal))).astype('float32')
    ###########################################################

    return signal_noise