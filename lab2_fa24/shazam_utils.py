import hashlib
import csv

from scipy.io import wavfile


HASHES_PER_PEAK = 15


def generate_hash(f1, f2, t1, t2, hash_length = 20):
    td = t2 - t1
    h = hashlib.sha1(("%s|%s|%s" % (str(f1), str(f2), str(td))).encode("utf-8"))
    return h.hexdigest()[0:hash_length], t1


def hashing(f1, t1, freq_indices, time_indices):
    freqs, times = f1[freq_indices], t1[time_indices]
    sorted_peaks = sorted(zip(freqs, times), key=lambda x: x[1])
    hashes = []

    for i in range(len(sorted_peaks)):
        for j in range(1, HASHES_PER_PEAK + 1):
            if i + j >= len(sorted_peaks):
                break
            f_1, t_1 = sorted_peaks[i]
            f_2, t_2 = sorted_peaks[i + j]
            hashes.append(generate_hash(f_1, f_2, t_1, t_2))

    return hashes
