
import numpy as np
import scipy.io
import scipy.signal
from random import random
from math import sin, cos, pi, sqrt
import matplotlib.pyplot as plt
import scipy.io.wavfile



class APS:
    
    

    data = None # All necessary data, need to load from Npy file 

    V_AIR = 34029  #Speed of air in cm/s
    LPF = 0 # Low Pass Filter
    beaconNum = 6    # Number of beacons in current system
    beaconList = None  # A list of beacons instance of Beacon Class
    rawSignal = None   # Raw signal, generated when microphone location is given
    carrierFreq = 12000 # Carrier frequency for carrier signal in order to generate modulated signal
    samplingRate = 44100 # Audio sampling rate
    offsetsPost = None  # Post processing offset
    distancesPost = None # Post processing distance    
    microphoneLocation = None # Microphone location, can be generated from "generate_microphone_loc" function
    ms = False
    
    
    def __init__(self, file_name, microphoneLocation = [25,30], testing = '', ms = False):
        
        
        
        """
            IMPORTANT: Run '%run -i support_code/APS.py' in jupyter notebook before use
            
            Default: file_name: 'new_data.npy', cleaned version of original MATLAB file with NPY extenstion (python friendly)
                     v_air = 34029 cm/s, can be changed to m/s by setting ms = True, the entire system in meter scale
                     microphoneLocation = [25,30], center of table
                     beaconsLocation: 1. In laboratory setting (ie: Entire system on the table, Coordinate =  array([[  0.,   0.], [ 53.,   3.],
                                                                                                                     [ 66.,  31.], [ 50.,  60.],
                                                                                                                     [ -4.,  58.], [-15.,  30.]])
                                                                                                                     
                                     2. Change to Testing setting (ie: Testing system for some functions, Coordinate =  array([[  0.,   0.], [500.,   0.],
                                                                                                                               [  0., 500.], [500., 500.],
                                                                                                                               [  0., 250.], [250.,   0.]])
                                        This can be done by setting testing = 'Test'
                                        
                    ms: 1. By default entire system in centimeter scale, 2 Change to meter scale by setting ms = True
                                                                                                               
            
            
            Instructions:
            IMPORTANT: Run '%run -i support_code/APS.py' in jupyter notebook before use
            1. Generate APS class instance: Lab = APS('new_data.npy')
            2. Generate/place microphone location in GUI: %matplotlib notebook
                                                          LOC = Lab.generate_microphone_loc()
            3. Generate raw signal with microphone location: Lab.generate_raw_signal(LOC, noise = True)
            4. Save the raw signal to a 'WAV' file: Lab.save_to_wav_file("APS_Sim_Class.wav")
            5. Post processing the "received" signal (aka. raw signal):
                a. For APS1: Lab.load_corr_sig()
                b. For APS2: Lab.simulation_testing("APS_Sim_Class.wav", construct_system, least_squares, isac=1)
        """
        
        
        
    # Loading data from npy file, and initialized original microphone location to be center of table
        
        self.data = np.load(file_name,allow_pickle=True).item()
        self.samplingRate = self.data['Fs']
        self.carrierFreq = self.data['f_c']
        self.V_AIR = self.data['v_air']
        self.LPF = self.data['LPF']
        self.microphoneLocation = np.array(microphoneLocation)
        self.beaconsLocation = self.data['beaconCoordinate'+testing]
        self.ms = ms
        
        if self.ms:  # m/s Mode
            self.V_AIR = self.V_AIR/100
            self.beaconsLocation = self.beaconsLocation/100
            self.microphoneLocation = self.microphoneLocation/100
                  
        # Generating a list of beacons 
        
        self.beaconList = []
        
        for i in range(self.beaconNum):
            
            self.beaconList.append(self.Beacons(self.beaconsLocation[i], 
                                           self.data['beacon%d'%i][0], 'Beacon%d'%i))


            
    def demodulate_signal(self, signal):
        """
        Demodulate the signal using complex demodulation.
        """
        # Demodulate the signal using cosine and sine bases
        demod_real_base = [cos(2 * pi * self.carrierFreq * i / self.samplingRate)
            for i in range(1, len(signal) + 1)]
        demod_imaginary_base = [sin(2 * pi * self.carrierFreq * i / self.samplingRate)
            for i in range(1, len(signal) + 1)]
        # Multiply the bases to the signal received
        demod_real = [demod_real_base[i] * signal[i] for i in range(len(signal))]
        demod_imaginary = [demod_imaginary_base[i] * signal[i] for i in range(len(signal))]
        # Filter the signals
        demod_real = np.convolve(demod_real, self.LPF)
        demod_imaginary = np.convolve(demod_imaginary, self.LPF)

        return np.asarray(demod_real) + np.asarray(demod_imaginary * 1j)
    
    
    def cross_correlation(self, stationary_signal, sliding_signal):
        """Compute the cross_correlation of two given signals    
        Args:
        stationary_signal (np.array): input signal 1
        sliding_signal (np.array): input signal 2

        Returns:
        cross_correlation (np.array): infinitely periodic cross-correlation of stationary_signal and sliding_signal

        >>> cross_correlation([0, 1, 2, 3], [0, 2, 3, 0])
        [8, 13, 6, 3]
        """
        inf_stationary_signal = np.concatenate((stationary_signal ,stationary_signal))
        entire_corr_vec = np.correlate(inf_stationary_signal, sliding_signal, 'full')
        return entire_corr_vec[len(sliding_signal)-1: len(sliding_signal)-1 + len(sliding_signal)]
    
    
    
    def generate_raw_signal(self, microphoneLocation, noise = False):
        
        self.microphoneLocation = microphoneLocation

        self.beaconList[0].generate_shifted_signal(microphoneLocation, self.V_AIR) 
    
        self.rawSignal = self.beaconList[0].shiftedSignal
        
        for b in self.beaconList[1:]:
            
            b.generate_shifted_signal(microphoneLocation,self.V_AIR)
            
            if b.offset == self.beaconList[0].offset:
                
                self.rawSignal += np.roll(b.shiftedSignal,1)
                
            else:
                
                self.rawSignal = self.rawSignal + b.shiftedSignal
        
        if noise:
#             print("Noise!!!!")
            noisySignal = self.add_random_noise(10)
            self.rawSignal = np.roll(np.tile(noisySignal, 10),2000)
        else:
#             print("No Noise!!!!")
            self.rawSignal = np.roll(np.tile(self.rawSignal, 10),2000)
        
        
        
    def add_random_noise(self, intensity):
        """
        Add noise to a given signal.
        Intensity: the Noise-Signal Ratio.
        """

        return self.rawSignal + np.random.rand(len(self.rawSignal))*intensity
    
    
        
    def save_to_wav_file(self, file_name):
        
        scipy.io.wavfile.write(file_name, self.samplingRate, self.rawSignal)
        
        
    def identify_peak(self, signal):
        """Returns the index of the peak of the given signal.
        Args:
        signal (np.array): input signal

        Returns:
        index (int): index of the peak

        >>> identify_peak([1, 2, 5, 7, 12, 4, 1, 0])
        4
        >>> identify_peak([1, 2, 2, 199, 23, 1])
        3
        """
        # YOUR CODE HERE
        return np.argmax(signal)






    def identify_offsets(self, averaged):
        """ Identify peaks in samples.
        The peaks of the signals are shifted to the center.

        Functions to Use: identify_peak

        Args:
        averaged (list): the output of the average_sigs() function (a list of single period cross correlated sigs)

        Returns (list): a list corresponding to the offset of each signal in the input
        """
        # Reshaping the input so that all of our peaks are centered about the peak of beacon0
        shifted = [np.roll(avg, len(averaged[0]) // 2 - self.identify_peak(averaged[0])) for avg in averaged]
        beacon0_offset = self.identify_peak(shifted[0])

        # shifted:
        #     All the signals shifted so beacon 0 is in the middle of the signal
        # beacon0_offset:
        #     The absolute index that beacon 0 starts on
        # YOUR TASK: Return a list that includes the relative indexes of where each beacon starts relative to
        #            when beacon 0 starts. This means that beacon 0's offset should be 0.

        # Use `shifted` and `beacon0_offset` to determine the offsets of other beacons
        # YOUR CODE HERE
        self.offsetsPost = np.array([self.identify_peak(i) - beacon0_offset for i in shifted])
        self.offsetsPost[1:][self.offsetsPost[1:] == 0] = 1


        



    def offsets_to_tdoas(self):
        """ Convert a list of offsets to a list of TDOA's

        Args:
        offsets (list): list of offsets in samples
        sampling_freq (int): sampling frequency in Hz

        Returns (list): a list of TDOAs corresponding to the input offsets
        """
        # YOUR CODE HERE
        tdoas = np.array([i/self.samplingRate for i in self.offsetsPost])
        return tdoas





    def signal_to_distances(self, t0):
        """ Returns a list of distances from the microphone to each speaker.

        Functions to use: offsets_to_tdoas, signal_to_offsets

        Args:
        raw_signal (np.array): recorded signal from the microphone
        t0 (float): reference time for beacon0 in seconds

        Returns (list): distances to each of the speakers (beacon0, beacon1, etc). in meters

        You will have to use v_air and sampling_freq above
        """
        # YOUR CODE HERE
        self.distancesPost = (np.array(self.offsets_to_tdoas()) + t0)*self.V_AIR
    
        



    
    
    def average_singular_signal(self, signal): 
        """ 
        Avarage over single signal for the length of one period
        """
        
        Lperiod = len(self.beaconList[0].binarySignal)
        Ncycle = len(signal) // Lperiod

        reshaped = signal.reshape((Ncycle,Lperiod))
        averaged = np.mean(np.abs(reshaped),0)

        return averaged



    

    def post_processing(self, signal):
        
        """
            UPDATE:  This fuction combines demodulate_signal, separate_signal and average_singular_signal together as post-processing function for
                the received data.
            
            
            
            ORIGINAL: Separate the beacons by computing the cross correlation of the raw signal
        with the known beacon signals.

        Args:
        raw_signal (np.array): raw signal from the microphone composed of multiple beacon signals

        Returns (list): each entry should be the cross-correlation of the signal with one beacon
        """
        Lperiod = len(self.beaconList[0].binarySignal)
        Ncycle = len(signal) // Lperiod
        
        demod = self.demodulate_signal(signal)
        
        cs = []
        avgs = []
        for ib, b in enumerate(self.beaconList):
            c = self.cross_correlation(demod[0:Lperiod],b.binarySignal)
            # Iterate through cycles
            for i in range(1,Ncycle):
                c = np.hstack([c, self.cross_correlation(demod[i*Lperiod:(i+1)*Lperiod], b.binarySignal)])
            cs.append(c)
            avg = self.average_singular_signal(c)      
            avgs.append(avg)
    
        return np.array(cs), np.array(avgs)

        
    def least_squares(self, A, b):
        """Solve the least squares problem
            
            Args:
            A (np.array): the matrix in the least squares problem
            b (np.array): the vector in the least squares problem
            
            Returns:
            pos (np.array): the result of the least squares problem (x)
            """
        
        # YOUR CODE HERE
        return  np.linalg.inv(A.T@A)@A.T@b
    
    
    def construct_system(self, speakers, TDOA, v_s):
        """Construct the components of the system according to a list of TDOA's
            Args:
            TDOA (np.array): an array of TDOA's
            
            Returns:
            A (np.matrix): the matrix corresponding to the least squares system
            b (np.array): the vector corresponding to the least squares system
            
        """
    
        A = np.zeros((speakers.shape[0]-1,3))
        A[:,:2] = speakers[1:]
        A[:,2] = (v_s**2)*TDOA[1:]
        A = A*2
        
        b = np.sum(speakers[1:,:] * speakers[1:,:],axis=1) - (v_s*TDOA[1:])**2
       
        return np.array(A), np.array(b)



    

    
    def generate_microphone_loc(self):
        """
        # Initialize center of the beacons and a round table
        """
        factor = 1
        if self.ms:
            factor = 0.01
        
        LOC = np.array([25,30], dtype = np.float32)*factor
        circle = plt.Circle(LOC, 48*factor, color='black', linewidth=7, fill=False)
        f,ax = plt.subplots(figsize=(10,10))
        xl = (-25*factor,75*factor)
        yl = (-20*factor,80*factor)
        
        
        def plot_table_beacons():
            """
             # Plot table and beacons
            """    
            plt.xlim(xl)
            plt.ylim(yl)
            colors = ['orange', 'g', 'c', 'y', 'm', 'b', 'k']
            for i,j in enumerate(self.beaconsLocation):
                plt.text(j[0]-5*factor,j[1]+1*factor,self.beaconList[i].name, color = colors[i], fontsize = 12)
                plt.scatter(j[0],j[1],marker='x', color = colors[i]) 
            plt.plot(25.280241935483872*factor, -17.958603896103895*factor,'o',color = 'black', markersize = 5, label = 'Table')

                
        def onclick(event):
            """
            # Feedback function for GUI
            """
            global LOC
            LOC = [event.xdata, event.ydata]

            plt.cla() 

            ax.add_artist(circle)
            plt.text(LOC[0]-4*factor,LOC[1]+1*factor,'Microphone',color = 'red')
            plt.scatter(LOC[0],LOC[1], color = 'red')
            plot_table_beacons()

            plt.legend()
            plt.show()

        """
        Initial Plot
        """
        plt.text(LOC[0]-15*factor,LOC[1]+2*factor,'Please place microphone within the table',color = 'red')
        c = plt.plot(LOC[0],LOC[1], marker='o', color = 'red', markersize = 10)
        plot_table_beacons()
        ax.add_artist(circle)
        plt.legend()
        plt.show()

        #     f.canvas.mpl_disconnect(eid)
        f.canvas.mpl_connect('button_press_event', onclick)

        return LOC
    
    
    def load_corr_sig(self, identify_peak = None):

        # Plot the received signal
        plt.figure(figsize=(18,4))
        plt.plot(self.rawSignal)
        # Convert the received signals into the format our functions expect
        demod = self.demodulate_signal(self.rawSignal)
        Lperiod = len(self.beaconList[0].binarySignal)
        Ncycle = len(demod) // Lperiod
        sig = []
        # Iterate through beacons
        for ib, b in enumerate(self.beaconList[:4]):
            s = self.cross_correlation(demod[0:Lperiod],b.binarySignal)
            # Iterate through cycles
            for i in range(1,Ncycle):
                s = np.hstack([s, self.cross_correlation(demod[i*Lperiod:(i+1)*Lperiod], b.binarySignal)])

            sig.append(s)
#             print(s)
        sig = [self.average_singular_signal(s) for s in sig]
#         print(sig)
        # Plot the cross-correlation with each beacon
        plt.figure(figsize=(18,4))
        for i in range(len(sig)):
            plt.plot(range(len(sig[i])), sig[i], label=self.beaconList[i].name)
        plt.legend()

#         Scale the x axis to show +/- 1000 samples around the peaks of the cross correlation
        if not identify_peak:
            peak_times = ([self.identify_peak(i) for i in sig])
        else:
            peak_times = ([identify_peak(i) for i in sig])
        plt.xlim(max(min(peak_times)-1000, 0), max(peak_times)+1000)

        plt.show()

        
    def plot_speakers(self, plt, coords, distances, xlim=None, ylim=None, circle=True, name = False):
        """Plots speakers and circles indicating distances on a graph.
        coords: List of x, y tuples
        distances: List of distance from center of circle"""
        colors = ['orange', 'g', 'c', 'y', 'm', 'b', 'k']
        label = [0,1,2,3,4,5]
        xs, ys = zip(*coords)
        fig = plt.gcf()

        for i in range(len(xs)):
#            plt.scatter(xs[i], ys[i], marker='x', color=colors[i], label='Speakers')
            plt.scatter(xs[i], ys[i], marker='x', color=colors[i])
            if name:
                plt.text(xs[i]-0.05*xlim,ys[i] + 0.01*ylim,self.beaconList[i].name, color = colors[i], fontsize = 12)
    

        if circle==True:
            for i, point in enumerate(coords):
                fig.gca().add_artist(plt.Circle(point, distances[i], facecolor='none',
                                                ec = colors[i]))
#        plt.legend(bbox_to_anchor=(1.4, 1))
        plt.axis('equal')
        if not name:
            if xlim: plt.xlim(*xlim)
            if ylim: plt.ylim(*ylim)

            
            

    def draw_hyperbola(self, p1, p2, d):
        """ hyperbola drawing function """

        p1=np.matrix(p1)
        p2=np.matrix(p2)
        pc=0.5*(p1+p2)
        p21=p2-p1
        d21=np.linalg.norm(p21)
        th=np.array(np.matrix(list(range(-49,50)))*pi/100) #radian sweep list
        a=d/2
        b=((d21/2)**2-(a)**2)**0.5
        x=a/np.cos(th)
        y=b*np.tan(th) #hyperbola can be represented as (d*sec(th),d*tan(th))
        p=np.vstack((x,y))
        m=np.matrix([[p21[0,0]/d21, -p21[0,1]/d21],[p21[0,1]/d21, p21[0,0]/d21]]) #rotation matrix
        vc=np.vstack((pc[0,0]*np.ones(99),pc[0,1]*np.ones(99))) #offset
        return np.transpose(m*p+vc)
    
    def calculate_position(self, least_squares, construct_system, speakers, TDOA):
        return least_squares(*construct_system(speakers, TDOA, self.V_AIR))



        
    def simulation_testing(self, filename, construct_system = None, least_squares = None, signal_to_tdoas = None):
        # LOAD IN SIMULATION DATA
        record_rate, raw_signal = scipy.io.wavfile.read(filename)
        # Get single channel
        if (len(raw_signal.shape) == 2):
            raw_signal = raw_signal[:,0]
        plt.figure(figsize=(16,4))
        plt.title("Raw Imported Signal")
        plt.plot(raw_signal)     

        _, averaged = self.post_processing(raw_signal)
    
        if signal_to_tdoas:
            demod = self.demodulate_signal(raw_signal)
            TDOA = np.array(signal_to_tdoas(demod))
        else:
        	self.identify_offsets(averaged)
        	self.signal_to_distances(0)
        	TDOA = self.offsets_to_tdoas()
        

        # Plot the averaged and separated output for each beacon
        
        fig = plt.figure(figsize=(12,6))
        for i in range(len(averaged)):
            plt.subplot(3,2,i+1)
            plt.plot(averaged[i])
            plt.title("Extracted Beacon %d"%i)
        plt.tight_layout()
            
        # Load Beaker Locations for Simulation
        
        simulation = self.beaconsLocation
#        print('simulation', simulation)
#        print('TDOA', TDOA)
        if not least_squares and not construct_system:
            x, y, t0 = self.calculate_position(self.least_squares, self.construct_system, simulation, TDOA)
        else:
            x, y, t0 = self.calculate_position(least_squares, construct_system, simulation, TDOA)
        
        #print( "Distance differences (m)): [%s]\n"%", ".join(["%0.4f" % d for d in distances]))
        print( "Least Squares Microphone Position: %0.4f, %0.4f, time-offset: %0.4f" % (x, y,t0))
        print( "Actual Microphone Position: %0.4f, %0.4f" % (self.microphoneLocation[0],
                                                                   self.microphoneLocation[1]))

        # Beacon Distances using signals to distances and t0
        dist_from_origin = np.linalg.norm([x, y])
        dist_from_speakers = [d + dist_from_origin for d in (self.distancesPost if not signal_to_tdoas else TDOA*self.V_AIR)]
        print( "Calcuated Distances from Beacons : [%s]\n"%", ".join(["%0.4f" % d for d in dist_from_speakers]))

        # Beacon Distances using calculated microphone position and Euclidean Norm
        # dist_from_speakers = [np.linalg.norm([x - x_s, y - y_s]) for (x_s, y_s) in self.beaconsLocation]
        # print( "Calcuated Distances from Beacons : [%s]\n"%", ".join(["%0.4f" % d for d in dist_from_speakers]))

        # Plot speakers and Microphone
        xmin = min(simulation[:,0])
        xmax = max(simulation[:,0])
        xrange = xmax-xmin
        ymin = min(simulation[:,1])
        ymax = max(simulation[:,1])
        yrange = ymax-ymin
        plt.figure(figsize=(15,15))
        plt.scatter(x, y, marker='o', color='r')
        plt.text(x-0.05*xrange, y+0.01*yrange, 'Microphone', color='r')
        self.plot_speakers(plt, simulation, [d for d in dist_from_speakers],xrange,yrange, circle=False, name = True)


        # Plot linear equations for LS
        if not least_squares and not construct_system:
            A, b = self.construct_system(simulation, TDOA, self.V_AIR) #for debugging
        else:
            A, b = construct_system(simulation, TDOA, self.V_AIR) #for debugging
        colors = ['orange', 'g', 'c', 'y', 'm', 'b', 'k']
        x2 = np.linspace(xmin-xrange*.2, xmax+xrange*.2, 1000)
        j=0;

        for i in range(len(b)):
            j=j+1
            y2 = [(b[i] - A[i][0]*xi -A[i][2]*t0) / A[i][1] for xi in x2]
            plt.plot(x2, y2, color=colors[j], label="Linear Equation " + str(j), linestyle='-')
            plt.xlim(xmin-xrange*.2, xmax+xrange*.2)
            plt.ylim(ymin-yrange*.2, ymax+yrange*.2)
#             plt.legend(bbox_to_anchor=(1.4, 1))
#             plt.legend()

        #for i in range(5):
        #    hyp= self.draw_hyperbola(simulation[i+1], simulation[0], self.distancesPost[i+1]) #Draw hyperbola
        #    plt.plot(hyp[:,0], hyp[:,1], color=colors[i+1], label='Hyperbolic Equation '+str(i+1), linestyle=':')

        plt.xlim(xmin-xrange*.2, xmax+xrange*.2)
        plt.ylim(ymin-yrange*.2, ymax+yrange*.2)
#         plt.legend(bbox_to_anchor=(1.6, 1))
        plt.legend()
        plt.show()
        
        
    def user_test(self, construct_system, least_squares, signal_to_tdoas):
        filename = input("Type filename (including the .wav): ")
        self.simulation_testing(filename, construct_system, least_squares, signal_to_tdoas)


        
    class Beacons:
        """
        Beacon Class(include all the properties of each beacon):
     
        All the properties in beacons class are pre-processing data 
        base on the distance from microphone to beacon.
        """
        coordinate = [] #Beacon location
        offset = 0 # The offset between microphone and beacon based on the distance from microphone to beacon
        binarySignal = [] # The original binary signal of beacons
        modulatedSignal = [] # The modulated signal
        shiftedSignal = [] # Modulated + shifted baseon the offset
        signalLength = 0 # signal length = length of binary signal = length of modulated signal = length of shifted signal
        carrierSingal = [] # Complex modualation of carrier signal
        carrierFreq = 12000 # Carrier frequency
        sampling_freq = 44100 # Sampling frequency
        name = None # Name of beacon
        

        def __init__(self, coordinate, bSignal, name):
            
            self.coordinate = coordinate
            self.binarySignal = bSignal
            self.signalLength = len(self.binarySignal)
            self.generate_carrier(self.signalLength)
            self.modulate_signal()
            self.name = name
            

            
            
        def distance(self, microphoneLocation):
            
            return np.sqrt((self.coordinate[0]-microphoneLocation[0])**2 + 
                                 (self.coordinate[1]-microphoneLocation[1])**2) 
        
        def generate_offset(self, microphoneLocation, v_air):
            
            t_diff = self.distance(microphoneLocation)/v_air
            
            self.offset = int(t_diff * self.sampling_freq)
            
        
        def modulate_signal(self):
            """
            Modulate a given signal. The length of both signals MUST
            be the same.
            """
            
            self.modulatedSignal =  [self.binarySignal[i] * self.carrierSingal[i] for i in range(self.signalLength)]
        
        
        def generate_shifted_signal(self, microphoneLocation, v_air):
            
            self.generate_offset(microphoneLocation, v_air)
            
            self.shiftedSignal = np.roll(self.modulatedSignal, self.offset)
            
                         
        def generate_carrier(self,signalLength, RANDOM_OFFSET = False):


            if RANDOM_OFFSET:
                rand = random()
                carrier_sample = (2 * pi *
                    (self.carrierFreq * sample / self.sampling_freq + rand)
                    for sample in range(1, signalLength + 1))           
            else: 
                carrier_sample = (2 * pi *
                    self.carrierFreq * sample / self.sampling_freq
                    for sample in range(1, signalLength + 1))

            self.carrierSingal = [cos(sample) for sample in carrier_sample]
            

