import numpy as np
from python_speech_features import mfcc

def pad(sequence, max_len):
    if len(sequence) < max_len:
        sequence = np.append(sequence, [0] * (max_len - len(sequence)))
    else:
        sequence = sequence[:max_len]
    return sequence

def partition_sequence(sequence, rate, window_size):
    sequence = pad(sequence, 16000)
    time_len = len(sequence) / rate
    partition_len, partitions = int(rate * window_size), []
    for i in range(0, len(sequence), partition_len):
        start, end = i, min(i + partition_len, len(sequence))
        partitions.append(sequence[start:end])
    return np.vstack(partitions)

def mfcc_sequence(sequence, rate, window_size):
    sequence = pad(sequence, 16000)
    return mfcc(sequence, samplerate=rate, winlen=window_size)
