import numpy as np
from utils.code_exercise1 import my_window
import matplotlib.pyplot as plt

def convert_to_samples(milliseconds: int, sampling_freq: int):
    """
    Convert a millisecond duration into the number of samples given the sampling frequency.

    :param milliseconds: duration to be converted to number of samples
    :param sampling_freq: the sampling frequency
    :return: number of samples
    """
    return int(milliseconds * (10 ** (-3)) * sampling_freq)



def compute_istft(stft: np.ndarray, sampling_rate: int, frame_shift: int, synthesis_window: np.ndarray) -> [np.ndarray]:
    """
    Compute the inverse short-time Fourier transform.

    :param stft: STFT transformed signal
    :param sampling_rate: the sampling rate in Hz
    :param frame_shift: the frame shift used to compute the STFT in milliseconds
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :return: a numpy array containing the time domain signal
    """

    # compute inverse rFFT and apply synthesis window
    time_frames = np.fft.irfft(stft)
    num_frames, samples_per_frame = time_frames.shape
    assert samples_per_frame == len(synthesis_window), "Synthesis window must match the number of samples per frame."
    time_frames *= synthesis_window

    # compute output size
    samples_per_shift = convert_to_samples(frame_shift, sampling_rate)
    output_len = samples_per_frame + (num_frames - 1) * samples_per_shift
    time_signal = np.zeros((output_len))


    # reconstruct signal by adding overlapping windowed segments
    for i in range(num_frames):
        time_signal[i*samples_per_shift:i*samples_per_shift+samples_per_frame] += time_frames[i]

    return time_signal

def compute_stft(v_signal: np.ndarray, sampling_rate: int, frame_length: int, frame_shift: int, v_analysis_window: np.ndarray):
    segments, v_time = my_window(v_signal, sampling_rate, frame_length, frame_shift)

    seg_windows = segments * v_analysis_window

    n_length = int(sampling_rate * frame_length / 1000)
    freqs = np.fft.fftfreq(n_length)

    m_stft = np.fft.fft(seg_windows)[:,:(n_length//2 +1)] # consider only lower half of the spectrum
    m_stft +=  np.finfo(np.float64).eps # avoid 0

    v_freq = freqs[:n_length//2] * sampling_rate

    return m_stft, v_freq, v_time


def plot_spectrum(m_stft, v_freq, v_time):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    im = ax.imshow(10* np.log10(np.maximum(np.square(np.abs(m_stft.T)), 10**( -15))),
                    cmap ='viridis', origin ='lower', extent =[ v_time [0] , v_time [-1] , v_freq [0] ,
                    v_freq [ -1]] , aspect ='auto')
    fig.colorbar(im , orientation ="vertical", pad =0.2)
