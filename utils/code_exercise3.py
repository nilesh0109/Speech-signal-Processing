import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.signal import freqz, lfilter

def get_a_vector(seg, m_length=12):
    autocorrelation = np.correlate(seg, seg, 'full')

    num_samples = seg.shape[0]
    zero_idx = num_samples-1
    M_idx = zero_idx+m_length+1

    lp_coeffs = solve_toeplitz(-autocorrelation[zero_idx:M_idx-1], autocorrelation[zero_idx+1:M_idx])

    return lp_coeffs

def get_b0(resp): # to adjust Hz
    energy = np.power(resp, 2).sum()
    b0 = np.sqrt(energy)
    return b0

def inspect_segment_lpc(segment, a, n_length, sr):
    freq, resp = freqz(1, np.concatenate(([1], a)), n_length, whole=True, fs=sr)

    S = np.fft.rfft(segment)

    resp = resp[:S.shape[0]]
    freq = freq[:S.shape[0]]

    S_db = 10* np.log10(np.abs(S))

    e_voiced = lfilter(np.concatenate(([1], a)), 1, segment)
    b0 = get_b0(e_voiced)
    estimated_signal = 10* np.log10(b0 * resp)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    ax.plot(freq[:S_db.shape[0]], estimated_signal[:S_db.shape[0]], label="b0 * H(z)")
    ax.plot(freq[:S_db.shape[0]], S_db, label="S DFT")
    ax.set_ylabel("dB")
    ax.set_xlabel("Hz")
    fig.legend()

    return fig
