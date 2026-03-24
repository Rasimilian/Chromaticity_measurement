from utils.naff_analysis import naff
import numpy as np


def find_three_peaks(array, start, stop):
    arr = array[start:stop]
    if len(arr) < 3:
        return []

    peaks = []

    if arr[0] > arr[1]:
        peaks.append((0, arr[0]))

    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append((i, arr[i]))

    if arr[-1] > arr[-2]:
        peaks.append((len(arr) - 1, arr[-1]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    peaks = [peak[0] + start for peak in peaks[:3]][::-1]
    peaks.sort(key=lambda x: x, reverse=False)
    return peaks


def get_spectrum(signal, nturns, freq_range):
    nturns = min(nturns, len(signal))
    main_freq, main_freq_amplitude, freq_spectrum, amplitude_spectrum = np.nan, np.nan, np.array([]), np.array([])
    try:
        freq, freq_spectrum, amplitude_spectrum, main_freq_amplitude = naff(signal, nturns, nterms=1, skipTurns=0, window=1, freq_range=freq_range)
        main_freq = freq[0][1] if freq[0][1] <= 0.5 else (1 - freq[0][1])
    except ValueError:
        print(f"Data size is less than turns {nturns}")
    return main_freq, main_freq_amplitude, freq_spectrum, amplitude_spectrum
