import numpy as np

def my_window(signal, sampling_rate, frame_length, frame_shift):
    m = []
    v = []

    n_length = int(sampling_rate * frame_length / 1000) # how many frames in 1 segment
    n_shift = int(sampling_rate * frame_shift / 1000) # how many frames to shift

    total_segments = 1 + (signal.shape[0] - n_length) // n_shift

    for i in range(total_segments):
        start = i * n_shift
        end = start + n_length

        center = (start + end) / 2
        period = center / sampling_rate

        v.append(period)
        m.append(signal[start:end])

    m = np.array(m)
    v = np.array(v)

    return m, v

def get_fundamental_freq(signal, sampling_rate):
    m, v = my_window(signal, sampling_rate, 32, 16)
    res = []

    for frame in m:
        corrs = np.convolve(frame, frame[::-1])
        corrs = corrs[corrs.shape[0]//2:] # remove negative lags

        lags = np.arange(corrs.shape[0]) # lags index
        periods = lags / sampling_rate
        freqs = np.ones(periods.shape[0]) / periods

        mask = ((freqs > 80.0) & (freqs < 400.0)) # mask to select freq ranging from 80 to 400

        filtered_corrs = np.zeros(corrs.shape[0])
        filtered_corrs[mask] = corrs[mask] # set out of range freq correlation to 0

        max_freq_idx = np.argmax(filtered_corrs) # take the idx where the correlation is maximum
        res.append(freqs[max_freq_idx]) # get the fundamental freq


    res = np.array(res)
    return res, v
