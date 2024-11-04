from __future__ import division
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# from IPython.html.widgets import *
from IPython.html.widgets import interact
#from ipywidgets import *
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt
from random import random
import scipy.io
import scipy.signal
from APS import APS


Lab = APS('new_data.npy', testing = 'Test', ms = True)


#########
# APS 1 #
#########

def cross_correlation(stationary_signal, sliding_signal):
    """Compute the cross_correlation of two given signals
    Args:
    stationary_signal (np.array): input signal 1
    sliding_signal (np.array): input signal 2

    Returns:
    cross_correlation (np.array): cross-correlation of stationary_signal and sliding_signal

    >>> cross_correlation([0, 1, 2, 3], [0, 2, 3, 0])
    [8, 13, 6, 3]
    """
    # new "infinitely periodic correletaion" using np.correlate like in HW
    inf_stationary_signal = np.concatenate((stationary_signal,stationary_signal))
    entire_corr_vec = np.correlate(inf_stationary_signal, sliding_signal, 'full')
    return entire_corr_vec[len(sliding_signal)-1: len(sliding_signal)-1 + len(sliding_signal)]
    # old implementation
    # return np.fft.ifft(np.fft.fft(stationary_signal) * np.fft.fft(sliding_signal).conj()).real
    
def cross_corr_demo():
    # Here we repeat the above example for a three-period case

    # Input signals for which to compute the cross-correlation
    # Make signals periodic with the numpy.tile function
    Nrepeat = 3
    signal1_base = np.array([1, 2, 3, 2, 1, 0])
    signal2_base = np.array([3, 1, 0, 0, 0, 1])
    zero_pad = np.zeros(len(signal2_base))
    signal2 = np.hstack((zero_pad, signal2_base, zero_pad)).astype(int) 
    signal1 = np.tile(signal1_base, Nrepeat)
    print('periodic stationary signal:'+ '[ ... ' + ' '.join([str(elem) for elem in signal1]) + ' ...]')
    print('sliding signal: '+str(signal2_base))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal1, np.roll(signal2,k)).astype(int)[0] for k in range(len(signal2))]
    print('periodic cross-correlation:'+ '[ ... ' + ' '.join([str(elem) for elem in corr]) + ' ...]')

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12, 24))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    #signal2 = np.roll(signal2, -len(signal2_base))
    
    for i in range(6, 12):
        plt.subplot(6,1,i-5)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal1, 'bo-', label='stationary')
        plt.plot(signal2, 'rx-', label='sliding')
        plt.xlim(0, 17)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation[%d]=%d\n%s\n%s'%(i, np.dot(signal1_base, signal2_base), str(signal2), str(signal1)))
        signal2 = np.roll(signal2, 1)
        signal2_base = np.roll(signal2_base, 1)

    # Adjust subplot spacing
    #plt.tight_layout()
    plt.figure(figsize=(12, 6))
    plt.plot(corr,'ko-')
    plt.xlim(0, 17)
    plt.ylim(0, 15)
    plt.title('Periodic Cross-correlation')

def cross_corr_demo_1():
    # Input signals for which to compute the cross-correlation
    signal1 = np.array([1, 2, 3, 2, 1, 0]*3) #inf periodic stationary
    signal2 = np.array([3, 1, 0, 0, 0, 1]) #sliding
    print('input sliding signal: '+str(signal2))
    print('input stationary signal: '+str(signal1))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal1, np.roll(signal2,k))[0] for k in range(len(signal2))]
    print('cross-correlation:'+str(corr))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(12,6))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal2, 'rx-', label='sliding')
        plt.plot(signal1, 'bo-', label='stationary')
        plt.xlim(0, 5)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal2), str(signal1)))
        signal2 = np.roll(signal2, 1)

    # Adjust subplot spacing
    #plt.tight_layout()
    plt.figure()
    plt.plot(corr,'ko-')
    plt.xlim(0, len(signal2)-1)
    plt.ylim(0, 15)
    plt.title('Cross-correlation (single-period)')

def cross_corr_demo_2():
    # Here we repeat the above example for a two-period case

    # Input signals for which to compute the cross-correlation
    # Make signals periodic with the numpy.tile function
    Nrepeat = 3
    signal1 = np.array([1, 2, 3, 2, 1, 0])
    signal2_base = np.array([3, 1, 0, 0, 0, 1])
    zero_pad = np.zeros(len(signal2_base))
    signal2 = np.hstack((zero_pad, signal2_base, zero_pad)).astype(int) 
    signal1 = np.tile(signal1, Nrepeat)
    print('input stationary signal: '+str(signal1))
    print('input sliding signal: '+str(signal2_base))

    # Use the numpy.roll function to shift signal2 in a circular way
    # Use the numpy.correlate function to convolute signal1 and signal2
    # Index [0] is used to convert a 1x1 array into a number
    corr = [np.correlate(signal1, np.roll(signal2,k)).astype(int)[0] for k in range(len(signal2))]
    print('cross-correlation:'+str(corr[6:12]))

    # Plot each operation required to compute the cross-correlation
    plt.figure(figsize=(36, 24))
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    #signal2 = np.roll(signal2, -len(signal2_base))
    
    for i in range(6, 12):
        plt.subplot(3,6,i+1)
        plt.subplots_adjust(hspace = 1);
        plt.plot(signal1, 'bo-', label='sliding')
        plt.plot(signal2, 'rx-', label='stationary')
        plt.xlim(0, 17)
        plt.ylim(0, 4)
        plt.legend(loc = 'upper left')
        #plt.title('Computed cross-correlation(%d)=%d\n%s\n%s'%(i, np.dot(signal1, signal2), str(signal2), str(signal1)))
        signal2 = np.roll(signal2, 1)

    # Adjust subplot spacing
    #plt.tight_layout()
    plt.figure()
    plt.plot(corr,'ko-')
    plt.xlim(6, 12)
    plt.ylim(0, 15)
    plt.title('Cross-correlation (two-period)')

def test_correlation_plot(signal1, signal2, lib_result, your_result):
    # Plot the output
    fig = plt.figure(figsize=(8,3))
    ax = plt.subplot(111)
    str_corr='Correct Answer (length='+str(len(lib_result))+')'
    str_your='Your Answer (length='+str(len(your_result))+')'

    ax.plot([x-len(signal2)+1 for x in range(len(lib_result))], lib_result, 'k', label=str_corr, lw=1.5)
    ax.plot([x-len(signal2)+1 for x in range(len(your_result))], your_result, '--r', label=str_your, lw = 3)
    ax.set_title("Cross correlation of:\n%s\n%s"%(str(signal1), str(signal2)))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def cross_corr_test():
    # You can change these signals to get more test cases
    # Test 1
    signal1 = np.array([1, 5, 8, 6])
    signal2 = np.array([1, 3, 5, 2])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)

    # Test 2
    signal1 = np.array([1, 5, 8, 6, 1, 5, 8, 6])
    signal2 = np.array([1, 3, 5, 2, 1, 3, 5, 2])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)

    # Test 3
    signal1 = np.array([1, 3, 5, 2])
    signal2 = np.array([1, 5, 8, 6])

    # Run the test
    lib_result, your_result = test_correlation(cross_correlation, signal1, signal2)
    test_correlation_plot(signal1, signal2, lib_result, your_result)


def test_correlation(cross_correlation, signal_one, signal_two):
#    result_lib = np.convolve(signal_one, signal_two[::-1])
    result_lib = np.array([np.correlate(signal_one, np.roll(signal_two, k)) for k in range(len(signal_two))])
    result_stu = cross_correlation(signal_one, signal_two)
    return result_lib, result_stu



def test(cross_correlation, identify_peak, test_num):
    # Virtual Test

    # Utility Functions
    def list_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if lst1[i] != lst2[i]: return False
        return True

    test_cases = {1: "Cross-correlation", 2: "Identify peaks", 3: "Arrival time"}

    # 1. Cross-correlation function
    # If you tested on the cross-correlation section, you should pass this test
    if test_num == 1:
        signal_one = [1, 4, 5, 6, 2]
        signal_two = [1, 2, 0, 1, 2]
        test = list_eq(cross_correlation(signal_one, signal_two), np.convolve(signal_one, signal_two[::-1]))
        if not test:
            print("Test {0} {1} Failed".format(test_num, test_cases[test_num]))
        else: print("Test {0} {1} Passed".format(test_num, test_cases[test_num]))

    # 2. Identify peaks
    if test_num == 2:
        test1 = identify_peak(np.array([1, 2, 2, 199, 23, 1])) == 3
        test2 = identify_peak(np.array([1, 2, 5, 7, 12, 4, 1, 0])) == 4
        your_result1 = identify_peak(np.array([1, 2, 2, 199, 23, 1]))
        your_result2 = identify_peak(np.array([1, 2, 5, 7, 12, 4, 1, 0]))
        if not (test1 and test2):
            print("Test {0} {1} Failed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your peaks [{2},{3}], Correct peaks [3,4]".format(test_num, test_cases[test_num], your_result1, your_result2))
    # 3. Virtual Signal
    if test_num == 3:
        transmitted = np.roll(beacon[0], -10) + np.roll(beacon[1], -103) + np.roll(beacon[2], -336)
        offsets = [0,0,0] #arrival_time(beacon[0:3], transmitted)
        test = (offsets[0] - offsets[1]) == (103-10) and (offsets[0] - offsets[2]) == (336-10)
        your_result1 = (offsets[0] - offsets[1])
        your_result2 = (offsets[0] - offsets[2])
        if not test:
            print("Test {0} {1} Failed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))
        else: print("Test {0} {1} Passed: Your offsets [{2},{3}], Correct offsets [93,326]".format(test_num, test_cases[test_num], your_result1, your_result2))



def test_identify_offsets(identify_offsets):
    # Utility Functions
    def list_float_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if abs(lst1[i] - lst2[i]) >= 0.00001: return False
        return True

    def list_sim(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if abs(lst1[i] - lst2[i]) >= 3: return False
        return True

    test_num = 0

    # 1. Identify offsets - 1
    print(" ------------------ ")
    test_num += 1
#     test_signal = get_signal_virtual(offsets = [0, 254, 114, 22, 153, 625])
    test_signal = np.load('test_identify_offsets1.npy')
#     raw_signal = demodulate_signal(test_signal)
#     sig = separate_signal(raw_signal)
    _,avgs = Lab.post_processing(test_signal)
    offsets = identify_offsets(avgs)
    test = list_sim(offsets, [0, 254, 114, 23, 153, 625])
    print("Test positive offsets")
    print("Your computed offsets = {}".format(offsets))
    print("Correct offsets = {}".format([0, 254, 114, 23, 153, 625]))
    if not test:
        print(("Test {0} Failed".format(test_num)))
    else:
        print("Test {0} Passed".format(test_num))

    # 2. Identify offsets - 2
    print(" ------------------ ")
    test_num += 1
#     test_signal = get_signal_virtual(offsets = [0, -254, 0, -21, 153, -625])
    test_signal = np.load('test_identify_offsets2.npy')
#     raw_signal = demodulate_signal(test_signal)
#     sig = separate_signal(raw_signal)
    _,avgs = Lab.post_processing(test_signal)
    offsets = identify_offsets(avgs)
    test = list_sim(offsets, [0, -254, 0, -21, 153, -625])
    print("Test negative offsets")
    print("Your computed offsets = {}".format(offsets))
    print("Correct offsets = {}".format([0, -254, 0, -21, 153, -625]))
    if not test:
        print("Test {0} Failed".format(test_num))
    else:
        print("Test {0} Passed".format(test_num))

def test_offsets_to_tdoas(offsets_to_tdoas):
    # 3. Offsets to TDOA

    def list_float_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if abs(lst1[i] - lst2[i]) >= 0.00001: return False
        return True

    print(" ------------------ ")
    test_num = 1
    off2t = offsets_to_tdoas([0, -254, 0, -21, 153, -625], 44100)
    test = list_float_eq(np.around(off2t,6), np.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6))
    print("Test TDOAs")
    print("Your computed TDOAs = {}".format(np.around(off2t,6)))
    print("Correct TDOAs = {}".format(np.around([0.0, -0.005759637188208617, 0.0, -0.0004761904761904762, 0.0034693877551020408, -0.01417233560090703],6)))
    if not test:
        print("Test Failed")
    else:
        print("Test Passed")




def test_signal_to_distances(signal_to_distances):
    def list_float_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if abs(lst1[i] - lst2[i]) >= 0.00001: return False
        return True
    # 4. Signal to distances
    print(" ------------------ ")
    test_num = 1
    Lab.generate_raw_signal([1.765, 2.683])
#     dist = signal_to_distances(demodulate_signal(get_signal_virtual(x=1.765, y=2.683)), 0.009437530220245524)
    signal = Lab.demodulate_signal(Lab.rawSignal)
    dist = signal_to_distances(signal , 0.009437530220245524)
    test = list_float_eq(np.around(dist,1), np.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1))
    print("Test computed distances")
    print("Your computed distances = {}".format(np.around(dist,1)))
    print("Correct distances = {}".format(np.around([3.2114971586473495, 4.1991869545657172, 2.9105604239534717, 3.9754134851779623, 1.7762604239534723, 2.7870991994636762],1)))
    if not test:
        print("Test Failed")
    else:
        print("Test Passed")

def test_signal_to_tdoas(signal_to_tdoas):
    def list_float_eq(lst1, lst2):
        if len(lst1) != len(lst2): return False
        for i in range(len(lst1)):
            if abs(lst1[i] - lst2[i]) >= 0.00001: return False
        return True
    print(" ------------------ ")
    test_num = 1
    Lab.generate_raw_signal([1.765, 2.683])
    signal = Lab.demodulate_signal(Lab.rawSignal)


    dist = signal_to_tdoas(signal)
    correct = np.around([0., 0.00290249, -0.00088435, 0.0022449, -0.00424036, -0.00126984], 10)
    test = list_float_eq(np.around(dist,10), correct)
    print("Test computed distances")
    print("Your computed distances = {}".format(np.around(dist,10)))
    print("Correct distances = {}".format(correct))
    if not test:
        print("Test Failed")
    else:
        print("Test Passed")





# Model the sending of stored beacons, first 2000 samples
sent_0 = Lab.beaconList[0].binarySignal[:2000]
sent_1 = Lab.beaconList[1].binarySignal[:2000]
sent_2 = Lab.beaconList[2].binarySignal[:2000]

# Model our received signal as the sum of each beacon, with some delay on each beacon.
delay_samples0 = 0;
delay_samples1 = 0;
delay_samples2 = 0;
received = np.roll(sent_0,delay_samples0) + np.roll(sent_1,delay_samples1) + np.roll(sent_2,delay_samples2)

def pltBeacons(delay_samples0,delay_samples1,delay_samples2):
    received_new = np.roll(sent_0,delay_samples0) + np.roll(sent_1,delay_samples1) + np.roll(sent_2,delay_samples2)
    plt.figure(figsize=(10,4))
    plt.subplot(2, 1, 1)
    plt.plot(received_new), plt.title('Received Signal (sum of beacons)'), plt.xlabel('Samples'), plt.ylabel('Amplitude')

    ax = plt.subplot(2, 1, 2)
    corr0 = cross_correlation(received_new, sent_0)
    corr1 = cross_correlation(received_new, sent_1)
    corr2 = cross_correlation(received_new, sent_2)
    plt.plot(range(-1000,1000), np.roll(corr0, 1000))
    plt.plot(range(-1000,1000), np.roll(corr1, 1000))
    plt.plot(range(-1000,1000), np.roll(corr2, 1000))
    plt.legend( ('Corr. with Beacon 0', 'Corr. with Beacon 1', 'Corr. with Beacon 2') )
    plt.title('Cross-correlation of received signal and stored copy of Beacon n')
    plt.xlabel('Samples'), plt.ylabel('Correlation')
    plt.tight_layout()
    plt.draw()

def sliderPlots():
	w0=widgets.IntSlider(min=-500, max=500, step=10, style={'description_width': 'initial'})
	w1=widgets.IntSlider(min=-500, max=500, step=10, style={'description_width': 'initial'})
	w2=widgets.IntSlider(min=-500, max=500, step=10, style={'description_width': 'initial'})
	interact(pltBeacons, delay_samples0 = w0, delay_samples1 = w1, delay_samples2 = w2)




def separate_signal(raw_signal):
    """Separate the beacons by computing the cross correlation of the raw signal
    with the known beacon signals.

    Args:
    raw_signal (np.array): raw signal from the microphone composed of multiple beacon signals

    Returns (list): each entry should be the cross-correlation of the signal with one beacon
    """
#     Lperiod = len(beacon[0])
#     Ncycle = len(raw_signal) // Lperiod
    Lperiod = len(Lab.beaconList[0].binarySignal)
    Ncycle = len(raw_signal) // Lperiod
    for ib, b in enumerate(Lab.beaconList):
#         print(raw_signal[0:Lperiod])
        c = cross_correlation(raw_signal[0:Lperiod],b.binarySignal)
        # Iterate through cycles
        for i in range(1,Ncycle):
            c = np.hstack((c, cross_correlation(raw_signal[i*Lperiod:(i+1)*Lperiod], b.binarySignal)))
        if (ib==0): cs = c
        else:       cs = np.vstack([cs, c])
    return cs

def average_multiple_signals(cross_correlations):
    Lperiod = len(Lab.beaconList[0].binarySignal)
    Ncycle = len(cross_correlations[0]) // Lperiod
    avg_sigs = []
    for c in cross_correlations:
        reshaped = c.reshape((Ncycle,Lperiod))
        averaged = np.mean(np.abs(reshaped),0)
        avg_sigs.append(averaged)

    return avg_sigs


def plot_average_multiple_signals(beacon_num=0):

    Lab.generate_raw_signal([1.2, 3.4])
    Lab.rawSignal = Lab.add_random_noise(25)
    cs, avgs = Lab.post_processing(np.roll(Lab.rawSignal,3000))
    period_len = Lab.beaconList[0].signalLength

    f, axarr = plt.subplots(2, sharex=True, sharey=True,figsize=(17,8))
    axarr[1].set(xlabel='Sample Number')
    period_ticks = np.arange(0, len(cs[0]), period_len)
    axarr[1].xaxis.set_ticks(period_ticks)

    axarr[0].plot(np.abs(cs[beacon_num]))
    [axarr[0].axvline(x=line, color = "red", linestyle='--') for line in period_ticks]
    axarr[0].set_title('Cross-Correlation of 2.5 sec Recording of Beacon 1 with Signal After Separation\n(No Averaging)')


    axarr[1].plot(avgs[beacon_num])
    axarr[1].axvline(x=period_ticks[0], color = "red", linestyle='--', label='period start')
    axarr[1].axvline(x=period_ticks[1], color = "red", linestyle='--')
    axarr[1].set_title('Averaged & Centered Periodic Cross-Correlation')
    plt.legend()

    stacked_cs = np.abs(cs[beacon_num])[:(len(cs[beacon_num])//period_len)*period_len].reshape(-1, period_len)
    print("Samples Offset of Each Period in Non-Averaged Signal:",[np.argmax(s) for s in stacked_cs])
    print("Samples Offset in Averaged Signal:",[np.argmax(avgs[beacon_num])])



def plot_shifted(identify_peak):
    # Simulate the received signal
#     test_signal = signal_generator(1.4, 3.22)
    Lab.generate_raw_signal([1.4, 3.22])
#     # Separate the beacon signals by demodulating the received signal
#     separated = separate_signal(test_signal)

#     # Perform our averaging function
#     avgs = average_multiple_signals(separated)
    _, avgs = Lab.post_processing(np.roll(Lab.rawSignal,-2500))

    # Plot the averaged output for each beacon
    plt.figure(figsize=(16,4))
    for i in range(len(avgs)):
        plt.plot(avgs[i], label="{0}".format(i))
    plt.title("Separated and Averaged Cross-correlation outputs with Beacon0 at t=0")
    plt.legend()
    plt.show()

    # Plot the averaged output for each beacon centered about beacon0
    plt.figure(figsize=(16,4))
    peak0 = identify_peak(avgs[0])
    Lperiod = len(avgs[0])
    for i in range(len(avgs)):
        plt.plot(np.roll(avgs[i], Lperiod//2 - peak0), label="{0}".format(i))
    plt.title("Shifted Cross-correlated outputs centered about Beacon0")
    plt.legend()
    plt.show()






##### for infinite periodic cross correlation plot ####
from IPython.html.widgets import interact, interactive, fixed, interact_manual
import IPython.html.widgets as widgets
from IPython.display import display

def correlation_plots(offset):
    stationary_coord = np.arange(-10,11)
    stationary_sig = np.array([-1, 0, 1, 0] * 5 + [-1])
    sliding_sig = np.array([-0.5, 0, 0.5, 0, -0.5])
    sliding_coord = np.array([-2,-1,0,1,2])
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(15,5))

    # plot stationary and sliding signal
    ax1.set_xlim(-10,10)
    ax1.plot(stationary_coord, stationary_sig, label = "periodic stationary signal")
    ax1.plot(sliding_coord+offset, sliding_sig, color="orange", label = "sliding signal")
    ax1.plot(np.arange(-10-8,-1)+offset, [0]*17, color="orange")
    ax1.plot(np.arange(2,11+8)+offset, [0]*17, color="orange")
    ax1.axvline(offset, color = "black", ls="--")
    ax1.set_xticks(np.arange(-10, 11, 1.0))
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend()

    # plot corr result
    corr = np.correlate(stationary_sig, sliding_sig, "full")[12-2:12+3]
    x = np.arange(-2,3,1)
    ax2.set_xlim(-10,10)
    ax2.set_ylim(-2, 2)
    ax2.plot(x, corr, label="periodic cross correlation", color="g")
    index = (offset+2)%4 - 2
    ax2.scatter(index, corr[index+2], color = "r")
    ax2.axvline(index, color = "black", ls="--")
    ax2.set_xticks(np.arange(-10, 11, 1.0))
    ax2.legend()
    ax2.set_title("cross_correlation([-1, 0, 1, 0, -1], [-0.5, 0, 0.5, 0, -0.5])")

    ax1.set_title("Periodic Linear Cross Correlation\nCorr Val at offset "+str(offset)+" is "+str(corr[index+2]))
    plt.show()

def inf_periodic_cross_corr():
    # interactive widget for playing with the offset and seeing the cross-correlation peak and aligned signals.
    widget = interactive(correlation_plots, offset=(-8, 8, 1))
    display(widget)




#########
# APS 2 #
########

def hyperbola_demo_1():
    # Assume we already know the time of arrival of the first beacon, which is R0/(speed_of_sound)
    labDemo = APS('new_data.npy', testing = 'Test', ms = True)
    labDemo.generate_raw_signal([1.2,3.6])
    _, separated = labDemo.post_processing(labDemo.rawSignal)
    labDemo.identify_offsets(separated)
    labDemo.signal_to_distances(((1.2)**2+(3.6)**2)**0.5/340.29)
    distances = labDemo.distancesPost[:4]
    print("The distances are: " + str(distances))
    TDOA = labDemo.offsets_to_tdoas()
    plt.figure(figsize=(8,8))
    dist=np.multiply(340.29,TDOA)
    colors = ['r', 'g', 'c', 'y', 'm', 'b', 'k']
    for i in range(3):
        hyp=labDemo.draw_hyperbola(labDemo.beaconsLocation[i+1], labDemo.beaconsLocation[0], dist[i+1]) #Draw hyperbola
        plt.plot(hyp[:,0], hyp[:,1], color=colors[i+1], label='Hyperbola for beacon '+str(i+1), linestyle='--')
    labDemo.plot_speakers(plt, labDemo.beaconsLocation[:4], distances)
    plt.xlim(-9, 18)
    plt.ylim(-6, 6)
    plt.legend()
    plt.show()

def plot_speakers_demo():
    # Plot the speakers
    plt.figure(figsize=(8,8))


    labDemo = APS('new_data.npy', testing = 'Test', ms = True)
    labDemo.generate_raw_signal([1.2,3.6])
    _, separated = labDemo.post_processing(labDemo.rawSignal)
    labDemo.identify_offsets(separated)
    labDemo.signal_to_distances(((1.2)**2+(3.6)**2)**0.5/340.29)
    distances = labDemo.distancesPost[:4]
    TDOA = labDemo.offsets_to_tdoas()
    v = labDemo.V_AIR


    # Plot the linear relationship of the microphone and speakers.
    isac=1; #index of the beacon to be sacrificed
    speakers = labDemo.beaconsLocation
    helper = lambda i: float(speakers[i][0]**2+speakers[i][1]**2)/(v*TDOA[i])-float(speakers[isac][0]**2+speakers[isac][1]**2)/(v*TDOA[isac])
    helperx = lambda i: float(speakers[i][0]*2)/(v*TDOA[i])-float(speakers[isac][0]*2)/(v*TDOA[isac])
    helpery = lambda i: float(speakers[i][1]*2)/(v*TDOA[i])-float(speakers[isac][1]*2)/(v*TDOA[isac])

    x = np.linspace(-9, 9, 1000)
    y1,y2,y3 = [],[],[]
    if isac!=1: y1 = [((helper(1)-helper(isac))-v*(TDOA[1]-TDOA[isac])-helperx(1)*xi)/helpery(1) for xi in x]
    if isac!=2: y2 = [((helper(2)-helper(isac))-v*(TDOA[2]-TDOA[isac])-helperx(2)*xi)/helpery(2) for xi in x]
    if isac!=3: y3 = [((helper(3)-helper(isac))-v*(TDOA[3]-TDOA[isac])-helperx(3)*xi)/helpery(3) for xi in x]

    # You can calculate and plot the equations for the other 2 speakers here.
    if isac!=1: plt.plot(x, y1, label='Equation for beacon 1', color='g')
    if isac!=2: plt.plot(x, y2, label='Equation for beacon 2', color='c')
    if isac!=3: plt.plot(x, y3, label='Equation for beacon 3', color='y')
    plt.legend()
    labDemo.plot_speakers(plt, labDemo.beaconsLocation[:4], distances)
    plt.legend(bbox_to_anchor=(1.4, 1))
    plt.xlim(-9, 11)
    plt.ylim(-6, 6)
    plt.show()

def construct_system_test(construct_system):


    labDemo = APS('new_data.npy', testing = 'Test', ms = True)
    labDemo.generate_raw_signal([1.2,3.6])
    _, separated = labDemo.post_processing(labDemo.rawSignal)
    labDemo.identify_offsets(separated)
    labDemo.signal_to_distances(((1.2)**2+(3.6)**2)**0.5/340.29)
    distances = labDemo.distancesPost[:4]
    TDOA = labDemo.offsets_to_tdoas()
    v = labDemo.V_AIR
    speakers = labDemo.beaconsLocation


    # Plot the linear relationship of the microphone and speakers.
    isac=1; #index of the beacon to be sacrificed
    A, b = construct_system(speakers,TDOA,labDemo.V_AIR)
    for i in range(len(b)):
        print ("Row %d: %.f should equal %.f"%(i, A[i][0] * 1.2 + A[i][1] * 3.6, b[i]))

def least_squares_test(least_squares):
    A = np.array(((1,1),(1,2),(1,3),(1,4)))
    b = np.array((6, 5, 7, 10))
    yourres = least_squares(A,b)
    print('Your results: ',yourres)
    correctres = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    print('Correct results: ',correctres)

# # Define a helper function to use least squares to calculate position from just the TDOAs
# def calculate_position(least_squares, construct_system, speakers, TDOA, v_s, isac=1):
#     return least_squares(*construct_system(speakers, TDOA, v_s, isac))

# Define a testing function
def test_loc(least_squares, construct_system, x_pos, y_pos, inten, debug=False):


    labDemo = APS('new_data.npy', testing = 'Test', ms = True)
    labDemo.generate_raw_signal([x_pos, y_pos])
    raw_signal = labDemo.add_random_noise(inten)
    _, avgs = labDemo.post_processing(labDemo.rawSignal)
    labDemo.identify_offsets(avgs)
    labDemo.signal_to_distances(((1.2)**2+(3.6)**2)**0.5/340.29)
    distances = labDemo.distancesPost[:4]
    TDOA = labDemo.offsets_to_tdoas()
    v = labDemo.V_AIR
    speakers = labDemo.beaconsLocation
    

    # Construct system of equations
    A, b = construct_system(speakers, TDOA, labDemo.V_AIR)

    # Calculate least squares solution
    pos = labDemo.calculate_position(least_squares, construct_system, speakers, TDOA)

    if debug:
        # Plot the averaged output for each beacon
        plt.figure(figsize=(12,6))
        for i in range(len(avgs)):
            plt.subplot(3,2,i+1)
            plt.plot(avgs[i])
            plt.title("Beacon %d"%i)
        plt.tight_layout()

        # Plot the averaged output for each beacon centered about beacon0
        plt.figure(figsize=(16,4))
        peak = labDemo.identify_peak(avgs[0])
        for i in range(len(avgs)):
            plt.plot(np.roll(avgs[i], len(avgs[0]) // 2 - peak), label="{0}".format(i))
        plt.title("Beacons Detected")
        plt.legend()
        plt.show()

        print( "Offsets (samples): %s"%str(labDemo.offsetsPost))
        print( "Times (s): [%s]\n"%", ".join(["%0.6f" % t for t in TDOA]))
        print( "Constructing system...")
        print( "Verifying system using known position...")
        for i in range(len(b)):
            print( "Row %d: %.f should equal %.f"%(i, A[i][0] * x_pos + A[i][1] * y_pos, b[i]))

        print( "\nCalculating least squares estimate...")
    print("Expected: (%.3f, %.3f); got (%.3f, %.3f)\n"%(x_pos, y_pos, pos[0], pos[1]))



