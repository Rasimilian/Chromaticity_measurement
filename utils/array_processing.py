import numpy as np

ARR_INDEX_TO_FILL = 5


def check_waveform_size(arr, target_length):
    if len(arr) > target_length:
        return arr[:target_length]
    elif len(arr) < target_length:
        fill_value = arr[-ARR_INDEX_TO_FILL]
        need_to_add = target_length - len(arr)
        return np.concatenate((arr, [fill_value] * need_to_add))
    else:
        return arr
