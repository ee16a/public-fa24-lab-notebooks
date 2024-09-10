import numpy as np
from scipy.io import wavfile
from scipy import signal


FS_AG, COLDPLAY_AG = wavfile.read("VivaLaVida.wav")
FS_AG, KILLERS_AG = wavfile.read("MrBrightside.wav")


def test_Q1a(centered_magnitude_spectrum):
    assert np.allclose(centered_magnitude_spectrum(np.ones(10)), np.array([ 0.,  0.,  0.,  0.,  0., 10.,  0.,  0.,  0.,  0.]))
    assert np.allclose(centered_magnitude_spectrum([10] * 5 + [6 * 2] + [3] * 4), np.array([ 2.,  9.,  5.46491887,  9., 21.6364198, 74., 21.6364198,  9.,  5.46491887,  9.]))
    x = np.array([0.77498361, 0.02466649, 0.5159193 , 0.75489558, 0.85777587, 0.56253859, 0.01691631, 0.65058933, 0.4358812, 0.80630942])
    assert np.allclose(centered_magnitude_spectrum(x), np.array([0.19752313, 1.15615072, 0.59527128, 1.55793533, 0.21334073, 5.4004757, 0.21334073, 1.55793533, 0.59527128, 1.15615072]))
    
    print('Question 1a Passed!')


def test_Q1c(compute_spectrogram):
    coldplay = np.mean(COLDPLAY_AG, axis=1)
    killers = np.mean(KILLERS_AG, axis=1)

    _, _, coldplay_spect_expected = signal.spectrogram(coldplay, FS_AG, nperseg=4096)
    _, _, killers_spect_expected = signal.spectrogram(killers, FS_AG, nperseg=4096)

    coldplay_spect_expected = 20 * np.log10(coldplay_spect_expected + 1e-12)
    killers_spect_expected = 20 * np.log10(killers_spect_expected + 1e-12)

    f1, t1, coldplay_spect = compute_spectrogram(FS_AG, coldplay)
    f2, t2, killers_spect = compute_spectrogram(FS_AG, killers)
    coldplay_spect = np.array(coldplay_spect)
    killers_spect = np.array(killers_spect)

    assert f1 is not None, 'coldplay array of sample frequencies was None'
    assert f2 is not None, 'killers array of sample frequencies was None'
    assert t1 is not None, 'coldplay array of segment times was None'
    assert t2 is not None, 'killers array of segment times was None'

    assert (coldplay_spect == coldplay_spect_expected).all(), 'coldplay spectrogram contained at least one incorrect value'
    assert (killers_spect == killers_spect_expected).all(), 'killers spectrogram contained at least one incorrect value'

    print('Question 1c Passed!')


def test_Q2a(freq_idx, time_idx):
    assert np.allclose(freq_idx[10:20], np.array([20, 22, 22, 24, 27, 40, 40, 45, 45, 53]))
    assert np.allclose(time_idx[30:40], np.array([132, 597, 784, 311, 421, 454, 357, 637, 271, 245]))

    print('Question 2a Passed!')


def test_Q2b(hashes):
    assert len(hashes) == 2970, 'incorrect number of hashes'
    assert hashes[:5] == [
        ('eb046cc61c9c72cbc3e9', 1.6853333333333333), 
        ('c3732d9a4bc79b82286e', 1.6853333333333333), 
        ('8ff87ca5f63e8a036f46', 1.6853333333333333), 
        ('793fe791fbba01ae5dac', 1.6853333333333333), 
        ('c466941e32a557ad2b59', 1.6853333333333333)
    ], 'last 5 hashes contained at least one invalid value'

    print('Question 2b Passed!')


def test_Q2c(fingerprint):
    assert np.allclose([pr[1] for pr in fingerprint(COLDPLAY_AG, FS_AG)[100:120]], [3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667])
    assert np.allclose([pr[1] for pr in fingerprint(COLDPLAY_AG, FS_AG, min_distance=10)[100:120]], [0.864, 0.864, 0.864, 0.864, 0.864, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334, 1.2373333333333334])
    assert np.allclose([pr[1] for pr in fingerprint(COLDPLAY_AG, FS_AG, amp_thresh=25)[100:120]], [3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 3.8506666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667, 4.746666666666667])
    assert np.allclose([pr[1] for pr in fingerprint(COLDPLAY_AG, FS_AG, hashes_per_peak=10)[100:120]], [5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 5.866666666666666, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667, 6.314666666666667])

    print('Question 2c Passed!')


def test_Q3a(get_20_second_segment):
    coldplay_20 = get_20_second_segment(FS_AG, COLDPLAY_AG)
    assert len(coldplay_20) == (20 * FS_AG), 'coldplay segment was not 20 seconds'
    assert np.isin(coldplay_20, COLDPLAY_AG).any(), 'coldplay segment content was not contained within coldplay audio'

    killers_20 = get_20_second_segment(FS_AG, KILLERS_AG)
    assert len(killers_20) == (20 * FS_AG), 'killers segment was not 20 seconds'
    assert np.isin(killers_20, KILLERS_AG).any(), 'killers segment content was not contained within killers audio'

    print('Question 3a Passed!')


def test_Q3b(basic_detect_test):
    assert basic_detect_test(FS_AG, KILLERS_AG) == ('MrBrightside.wav', 100.0), 'killers was not identified or confidence was not 100%'
    assert basic_detect_test(FS_AG, COLDPLAY_AG) == ('VivaLaVida.wav', 100.0), 'coldplay was not identified or confidence was not 100%'

    print('Question 3b Passed!')


def test_Q3c(gaussian_noise_detect_test):
    killers_fn, killers_confidence = gaussian_noise_detect_test(KILLERS_AG)
    assert killers_fn == 'MrBrightside.wav', 'killers was not identified'
    assert killers_confidence > 50, 'killers confidence is less than 50%'

    coldplay_fn, coldplay_confidence = gaussian_noise_detect_test(COLDPLAY_AG)
    assert coldplay_fn == 'VivaLaVida.wav', 'coldplay was not identified'
    assert coldplay_confidence > 50, 'coldplay confidence is less than 50%'

    print('Question 3c Passed!')
