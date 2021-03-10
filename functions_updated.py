## @author: Eduarda Centeno
#  Documentation for this module.
#
#  Created on Wed Feb  6 15:06:12 2019; -*- coding: utf-8 -*-; 


#################################################################################################################################
#################################################################################################################################
# This code was built with the aim of allowing the user to work with Spike2 .smr files and further perfom correlation analyses ##
# between specific acoustic features and neuronal activity.                                                                    ##
# In our group we work with Zebra finches, recording their neuronal activity while they sing, so the parameters here might     ##
# have to be ajusted to your specific data.                                                                                    ##                                                                                                                              ##
#################################################################################################################################
#################################################################################################################################

### Necessary packages
import neo
import nolds
import numpy as np
import pylab as py
import matplotlib.lines as mlines
import datetime
import os
import pandas
import scipy.io
import scipy.signal
import scipy.stats
import scipy.fftpack
import scipy.interpolate
from sklearn.linear_model import LinearRegression
import random
from statsmodels.tsa.stattools import acf
from matplotlib.ticker import FormatStrFormatter
import math



### Example of files that one could use:
"""
 file="CSC1_light_LFPin.smr" #Here you define the .smr file that will be analysed
 songfile="CSC10.npy" #Here you define which is the file with the raw signal of the song
 motifile="labels.txt" #Here you define what is the name of the file with the motif stamps/times

"""

## Some key parameters:
fs=32000.0 #Sampling Frequency (Hz)
n_iterations=1000 #For bootstrapping
window_size=100 #For envelope
lags=100 #For Autocorrelation
alpha=0.05 #For P value
premot=0.05 # Premotor window
binwidth=0.02 # Bin PSTH
sybs=[]
sybs_rBOS=[]
sybs_OCS=[]
idx_noisy_syb=-1
len_motif=0
len_OCS=0
noise_p="n" #noise present?
pos_syls_PSTH=[]
pos_syls_rBOS_PSTH=[]
pos_syls_OCS_PSTH=[]
#For spectrogram
window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"
smooth_win=10
##
#
# Fills the varaiables sybs, idx_noisy_syb, len_motif, noise_p. sybs contains the syllbales of the motif and possibly at the end the label of the noisy syllable. 
# idx_noisy_syb is the index in sybs of the syllable receiving noise. ex if motif is 'a','b','c' and 'c' is sometimes receiving noise and noisy 'c' is labeled 'd' then 
# sybs = ['a','b','c','d'] and idx_noisy_syb is 2, len_motif is 3
def read_syllable_list(syllable_list):
    global sybs
    global idx_noisy_syb
    global len_motif
    global noise_p
    #Is there song, with or without noise, is there playback
    syllables= open(syllable_list, "r")
    syllablesr= syllables.read().splitlines()
    idx_noisy_syb=int(syllablesr[1])
    len_motif=int(syllablesr[0])-1
    if(idx_noisy_syb==-1):
       noise_p="n"
    else:
       noise_p="y"
    for i in range(len_motif+1):
        sybs.append(syllablesr[i+2])
	
##
#
# Resets the vars
def reset_vars():
    global sybs
    global idx_noisy_syb
    global len_motif
    global noise_p
    global pos_syls_PSTH
    sybs=[]
    idx_noisy_syb=-1
    len_motif=0
    noise_p="n"
    pos_syls_PSTH=[]

##
#
# Fills the varaiables sybs, idx_noisy_syb, len_motif, noise_p. sybs contains the syllbales of the motif and possibly at the end the label of the noisy syllable. 
# idx_noisy_syb is the index in sybs of the syllable receiving noise. ex if motif is 'a','b','c' and 'c' is sometimes receiving noise and noisy 'c' is labeled 'd' then 
# sybs = ['a','b','c','d'] and idx_noisy_syb is 2, len_motif is 3
def read_playback_list(playback_list):
    global sybs_OCS
    global sybs_rBOS
    global len_OCS
    #Is there song, with or without noise, is there playback
    syllables= open(playback_list, "r")
    syllablesr= syllables.read().splitlines()
    for i in range(len_motif):
        sybs_rBOS.append(syllablesr[i])
    len_OCS=int(syllablesr[len_motif])
    for i in range(len_OCS):
        sybs_OCS.append(syllablesr[len_motif+1+i])
		
		
##
#
# Fills the variable song_PSTH. Contains the coordinates of the positions of the syllable labels on the PSTH
def initialize_song_PSTH(song_PSTH):
    global pos_syls_PSTH
    #Is there song, with or without noise, is there playback
    PSTH_params= open(song_PSTH, "r")
    PSTH_paramsr= PSTH_params.read().splitlines()
    print(len_motif)
    for i in range(len_motif):
        pos_syls_PSTH.append(PSTH_paramsr[i])	
		
##
#
# Fills the variable song_PSTH. Contains the coordinates of the positions of the syllable labels on the PSTH
def initialize_playback_PSTH(playback_PSTH):
    global pos_syls_rBOS_PSTH
    global pos_syls_OCS_PSTH
    #Is there song, with or without noise, is there playback
    PSTH_params= open(playback_PSTH, "r")
    PSTH_paramsr= PSTH_params.read().splitlines()
    for i in range(len_motif):
        pos_syls_rBOS_PSTH.append(PSTH_paramsr[i])
    for i in range(len_OCS):
        pos_syls_OCS_PSTH.append(PSTH_paramsr[len_motif+i])

	
def print_vars():
    print(sybs)
    print(idx_noisy_syb)
    print(len_motif)
    print(noise_p)
    print(pos_syls_PSTH)
	
def print_pb_vars():
    print(sybs_OCS)
    print(sybs_rBOS)
    print(len_OCS)
		

## 
#
# sybs taken from read_syllable_list(). Outputs a list k. k[i] is an array of (1,2) elements with onset,offset times 
# of a rendition of the ith syllable. Not to be used for playback psth except syllable by syllable psth (i.e. psth_pb())!!!                                                                                                          #
def sortsyls(motifile,n):
    #Read and import files that will be needed
    f=open(motifile, "r")
    imported = f.read().splitlines()
    
    if(n==0): #song
       sybs_=sybs # is always of the form ['a','b','c','d']where the last element 'd' is the noisy version of a syllable 'a' or 'b' or 'c'
    elif(n==1): #BOS
       sybs_=sybs[:-1] #last element is the potential noisy version of the syllable targeted with noise 
    elif(n==2):#==2, #rBOS
       sybs_=sybs_rBOS
    else:#==3 #OCS
       sybs_=sybs_OCS
	   
    nb_syls=len(sybs_)

    k=[]
    for i in range(nb_syls):
        k.append(np.empty((1,2)))
    #Excludes everything that is not a real syllable
    for i in range(len(imported)):
        syl=imported[i][-1]
        not_found=True
        idx_syl=0#idx of syl in the sybs_
        while(not_found and idx_syl<nb_syls): 
            not_found = not(syl==sybs_[idx_syl])
            idx_syl+=1
        if(not(not_found)):
            idx_syl=idx_syl-1	
            pars=[imported[i].split(",")]
            k[idx_syl]=np.append(k[idx_syl],np.array([int(pars[0][0]), int(pars[0][1])], float).reshape(1,2), axis=0)

    for i in range(nb_syls):
        k[i]=(k[i])[1:]

    finallist=[]
    for i in k:
        print(i.size)
        if i.size != 0:
            finallist+=[i]
        else:
            continue
    #print(finallist)
    return finallist


## 
#
# This function outputs the list of basebeg_pb and basend_pb needed for playback psth	
def baseline_playback(motifile):	
    #Read and import files that will be needed
    f=open(motifile, "r")
    imported = f.read().splitlines()
    
    baseline_pb=[[],[]]
    for i in range(1,len(imported)):
        if ((imported[i][-1] == "a")or(imported[i][-1] == "s")or(imported[i][-1] == "j")or(imported[i][-1] == "z")):
            offsets_i_1=[imported[i-1].split(",")]
            onsets_i=[imported[i].split(",")]			
            if((float(onsets_i[0][0])-float(offsets_i_1[0][1]))>2.1*fs): #at least 2 sec diff between offset of prev syll/noise and onset of actu syll
               (baseline_pb[0]).append(float(onsets_i[0][0])-2.1*fs)
               (baseline_pb[1]).append(float(onsets_i[0][0])-0.1*fs)

    baseline_pb	=np.asarray(baseline_pb)
    return baseline_pb		
	

## 
#
# This function outputs the list of onset/offsets of the white noise
def sortsyls_wn(motifile):
    #Read and import files that will be needed
    f=open(motifile, "r")
    imported = f.read().splitlines()
    
    #Excludes everything that is not a real syllable
    w=[]
    arrw=np.empty((1,2))
    for i in range(len(imported)):
        if imported[i][-1] == "z":
            w=[imported[i].split(",")]
            arrw=np.append(arrw, np.array([int(w[0][0]), int(w[0][1])], float).reshape(1,2), axis=0)
            
    arrw=arrw[1:]
    finallist=arrw
    return finallist

## 
#
# This  function will allow you to get onset,offfset times of the syllables within a (possibly not completed) motif
# n selects whether we deal with song, BOS, rBOS, or OCS
def sortsyls_psth_glob(motifile,n):
    #Read and import files that will be needed
    #sybs_pb=["j","k","l","d"]
    #sybs_pb=["a","b","c","d"] #labels of relevant syllables (syllables in the song motif), the last syllable is by convention the noisy syllable (one of the relevant syllables having received w noise)
    #sybs_pb=["s","t","u","v","w","d"]
    #idx_noisy_syb=2 #idex in sybs_pb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    f=open(motifile, "r")
    imported = f.read().splitlines()
	
    if(n==0): #song
       sybs_=sybs.copy() #is always of the form ['a','b','c','d']where the last element 'd' is the noisy version of a syllable 'a' or 'b' or 'c'
    elif(n==1): #BOS
       sybs_=sybs.copy() #is always of the form ['a','b','c','d']where the last element 'd' is the noisy version of a syllable 'a' or 'b' or 'c'
    elif(n==2):#==2, #rBOS
       sybs_=sybs_rBOS.copy()
       sybs_.append(sybs[-1]) #is always of the form ['j','k','l','d']where the last element 'd' is here for coherency
    else:#==3 #OCS
       sybs_=sybs_OCS.copy()
       sybs_.append(sybs[-1]) #is always of the form ['s','t','u','v','w','d']where the last element 'd' is here for coherency

	
    len_motif_=len(sybs_)-1 #length of the motif
    if((len_motif_-1)==idx_noisy_syb):
       	nb_chunk_psth=3 #abcd, abcd(n), abc. Only the last syll is authorized not to be sung
    else:
        nb_chunk_psth=4 #abcd, abc(n)d, abc, anc(n). Only the last syll is authorized not to be sung
		
    chunks=[]
    for i in range(2): #with or without noise
        chunks.append([])
		
    current_motif=(-1)*np.ones(2*len_motif_) #onsets and offsets of the syllables. If -1, not relevant number put in case the syll is not sung

    buffer=[[],[],[]] #onset, offset, label
    for i in range(len_motif_):
        buffer[0].append(0) #onset
        buffer[1].append(0) #offset
        buffer[2].append("") #label


    for i in range(len(imported)-len_motif_+1):
        #Define test variable:
        if(idx_noisy_syb!=0):
          test_var=(imported[i][-1] ==sybs_[0])
        else:
          test_var=((imported[i][-1] ==sybs_[0])or(imported[i][-1] ==sybs_[-1]))

        if (test_var): #first syllable of the motif, check type of motif:
            #print("syb a found")
            current_motif=(-1)*np.ones(2*len_motif_)
            for j in range(len_motif_):
                a=[imported[i+j].split(",")] 
                buffer[0][j]=int(a[0][0]) #onset
                buffer[1][j]=int(a[0][1]) #offset
                #buffer[2][j]=imported[i+j][-1] #label
                buffer[2][j]=a[0][2] #label
            #print(buffer)
            
            #check if song motif caught
            match = 10*(nb_chunk_psth) #30 or 40
            for j in range(len_motif_-1): #last syl is allowed not to be sung
                if(j!=idx_noisy_syb):
                   if(buffer[2][j]!=sybs_[j]):
                      match=0
                      break
                else: #idx of noisy syllable
                   if((buffer[2][j]!=sybs_[j])and(buffer[2][j]!=sybs_[-1])):
                      match=0
                      break
                   elif(buffer[2][j]==sybs_[-1]): #noisy syllable
                      match=match+2
						  
            #treat last syl
            if((buffer[2][len_motif_-1]!=sybs_[len_motif_-1])and(buffer[2][len_motif_-1]!=sybs_[-1])): #last syl not sung
               match=match-1
               #print("last not sung")
            elif(buffer[2][len_motif_-1]==sybs_[-1]): #last sung with noise
               match=match+2
               #print("last sung with noise")
			   
			   
            #extract onset,offset of the syllables in the motif (the motif can be incomplete, in which case there is a -1 in place of onset/offset)
			#abcd,abcd(n),abc
            if(match==30):
               for j in range(len_motif_):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset
               chunks[0].append(current_motif)
				   
            if(match==32):
               for j in range(len_motif_):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset
               chunks[1].append(current_motif)
				   
            if(match==29):
               for j in range(len_motif_-1):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset
               chunks[0].append(current_motif)
          
			
			#abcd,abc(n)d,abc,abc(n)
            if(match==40):
               for j in range(len_motif_):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset
               chunks[0].append(current_motif)
				   
            if(match==42):
               for j in range(len_motif_):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset
               chunks[1].append(current_motif)

            if(match==39):  
               for j in range(len_motif_-1):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset	
               chunks[0].append(current_motif)
				   
            if(match==41):  
               for j in range(len_motif_-1):
                   current_motif[2*j]=buffer[0][j]#onset
                   current_motif[2*j+1]=buffer[1][j]#offset	
               chunks[1].append(current_motif) 
			   
    #print("chunks:\n")
    #print(chunks)
    return chunks
	
def tellme(s):
    print(s)
    py.title(s, fontsize=10)
    py.draw()


def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=(500, 15900)):
    """filter song audio with band pass filter, run through filtfilt
    (zero-phase filter)

    Parameters
    ----------
    rawsong : ndarray
        audio
    samp_freq : int
        sampling frequency
    freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter.
        If None, no cutoffs; filtering is done with cutoffs set
        to range from 0 to the Nyquist rate.
        Default is [500, 10000].

    Returns
    -------
    filtsong : ndarray
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError('Low frequency cutoff {} is invalid, '
                         'must be greater than zero.'
                         .format(freq_cutoffs[0]))

    Nyquist_rate = samp_freq / 2
    if freq_cutoffs[1] >= Nyquist_rate:
        raise ValueError('High frequency cutoff {} is invalid, '
                         'must be less than Nyquist rate, {}.'
                         .format(freq_cutoffs[1], Nyquist_rate))

    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    #print("b is below")
    #print(b)
    #np.savetxt('b_20kHz.txt', b, fmt='%1.6f', delimiter='/n')
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    #filtsong = filter_song(b, a, rawsong)
    return (filtsong)
	
def smoothed(inputSignal,fs=fs, smooth_win=10):
        squared_song = np.power(inputSignal, 2)
        len = np.round(fs * smooth_win / 1000).astype(int)
        h = np.ones((len,)) / len
        smooth = np.convolve(squared_song, h)
        offset = round((smooth.shape[-1] - inputSignal.shape[-1]) / 2)
        smooth = smooth[offset:inputSignal.shape[-1] + offset]
        smooth = np.sqrt(smooth)
        return smooth
 

def smooth_data(rawsong, samp_freq, freq_cutoffs=None, smooth_win=10):

    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)
        #filtsong = rawsong

    filtsong=filtsong.astype(np.float)
	
    squared_song = np.power(filtsong, 2)

    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    #np.savetxt('h_20kHz.txt', h, fmt='%1.6f', delimiter='/n')
	#np.convolve uses (M,) like arrays
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth


#Fast loop to check visually if the syllables are ok. I've been finding problems in A syllables, so I recommend checking always before analysis.
def checksyls(songfile,motifile, beg, end):
    finallist=sortsyls(motifile,0)  
    song=np.load(songfile)
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=finallist[0]
    elif answer.lower() == "b":
        used=finallist[1]
    elif answer.lower() == "c":
        used=finallist[2]    
    elif answer.lower() == "d":
        used=finallist[3]
    
    print("This syb has "+ str(len(used)) + " renditions.")
    
    for i in range(beg,end):
        py.figure()
        py.plot(song[int(used[i][0]):int(used[i][1])])

""" The two following functions were obtained from 
http://ceciliajarne.web.unq.edu.ar/investigacion/envelope_code/ """
def window_rms(inputSignal, window_size=window_size):
        a2 = np.power(inputSignal,2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, "valid"))
    
def getEnvelope(inputSignal, window_size=window_size):
# Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = window_size # change this number depending on your signal frequency content and time scale
    outputSignal = []

    for baseIndex in range (0, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal

def jumpsyl(spikefile):
    with open("CheckSylsFreq"+spikefile[:-4]+".txt", "r") as datafile:
        fich=datafile.read().split()[1::4] 
        #print(fich)		
    return fich
###############################################################################################################################





##############################################################################################################################
# From now on there will be the core functions of this code, which will be individually documented:                          #
                                                                                                                             #

##
##
# 
# This  function will allow you to read the .smr files from Spike2.
def read(file):
    reader = neo.io.Spike2IO(filename=file) #This command will read the file defined above
    #reader = neo.io.RawBinarySignalIO(filename=file)
    data = reader.read()[0] #This will get the block of data of interest inside the file
    data_seg=data.segments[0] #This will get all the segments
    #an_sig=data_seg.analogsignals
    #print((data_seg.analogsignals[0].as_array()).shape)
    #print(data_seg.unit_channels[0])

    return data, data_seg


	
## 
#
# This  function will allow you to get information inside the .smr file.
# It will return the number of analog signals inside it, the number of spike trains, 
# a numpy array with the time (suitable for further plotting), and the sampling rate of the recording.
def getinfo(file):
    data, data_seg= read(file)
     # Get the informations of the file
    t_start=float(data_seg.t_start) #This gets the time of start of your recording
    #t_stop=float(data_seg.t_stop) #This gets the time of stop of your recording
    as_steps=len(data_seg.analogsignals[0]) 
    #time=np.linspace(t_start,t_stop,as_steps)
    n_analog_signals=len(data_seg.analogsignals) #This gets the number of analogical signals of your recording
    n_spike_trains=len(data_seg.spiketrains) #This gets the number of spiketrains signals of your recording
    ansampling_rate=int(data.children_recur[0].sampling_rate) #This gets the sampling rate of your recording
    t_stop=t_start+float(as_steps*ansampling_rate)
    time=np.linspace(t_start,t_stop,as_steps)
    return n_analog_signals, n_spike_trains, time, ansampling_rate
	

def getsong(file):
    data, data_seg = read(file)
    for i in range(len(data_seg.analogsignals)):
        if data_seg.analogsignals[i].name == 'Channel bundle (CSC10) ':
            song=data_seg.analogsignals[i].as_array()
        elif data_seg.analogsignals[i].name == 'CSC10':
              song=data_seg.analogsignals[i].as_array()
        else:
             continue
    np.save("Songfile", song)
## 
#
# This  function will get the analogical signals and the spiketrains from the .smr file and return them in the end as arrays.
def getarrays(file): #Transforms analog signals into arrays inside list
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    # Extract analogs and put each array inside a list
    analog=[]
    for i in range(n_analog_signals):
        analog += [data_seg.analogsignals[i].as_array()]
    print("analog: This list contains " + str(n_analog_signals) + " analog signals!")
    # Extract spike trains and put each array inside a list
    sp=[]
    for k in range(n_spike_trains):
        sp += [data_seg.spiketrains[k].as_array()]
    print("sp: This list contains " + str(n_spike_trains) + " spiketrains!")
    return analog, sp


## 
#
# This  function will allow you to plot the analog signals and spiketrains inside the .smr file.
def plotplots(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    #Plot of Analogs
    py.figure()
    for i in range(n_analog_signals):
        py.subplot(len(analog),1,i+1)
        py.plot(time,analog[i])
        py.xlabel("time (s)")
        py.ylabel("Amplitude")
        py.title("Analog signal of: " + data_seg.analogsignals[i].name.split(" ")[2])
    py.tight_layout()
    #Plot of Spike Trains
    Labels=[]
    for i in range(n_spike_trains):
        Chprov = data.list_units[i].annotations["id"]
        Labels += [Chprov]
    py.figure()
    py.yticks(np.arange(0, 11, step=1), )
    py.xlabel("time (s)")
    py.title("Spike trains")
    py.ylabel("Number of spike trains")
    res=-1
    count=0
    for j in sp:
        colors=["black","blue", "red", "pink", "purple", "grey", "limegreen", "aqua", "magenta", "darkviolet", "orange"] #This was decided from observing which order SPIKE2 defines the colors for the spiketrains
        res=res+1
        py.scatter(j,res+np.zeros(len(j)),marker="|", color=colors[count])
        py.legend((Labels), bbox_to_anchor=(1, 1))
        count+=1        
    py.tight_layout()
    py.show()

	
## 
#
# This function will create a few files inside a folder which will be named according to the date and time. Works with old version of neo
# The files are:
#
#  1 - summary.txt : this file will contain a summary of the contents of the .smr file.
#
#  2- the spiketimes as .txt: these files will contain the spiketimes of each spiketrain.
#
#  3 - the analog signals as .npy: these files will contain the raw data of the analog signals.   
#  
#  file is the .smr file of the recording
def createsave(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    #Create new folder and change directory
    today= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(today)
    os.chdir(os.path.expanduser(today))
    
    #Create DataFrame (LFP should be indicated by the subject) and SpikeTime files
    res=[]
    LFP = input("Enter LFP number:")
    print(data)
    if os.path.isfile("..//unitswindow.txt"):
        for i in range(n_spike_trains):
            Chprov = data_seg.spiketrains[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            with open("..//unitswindow.txt", "r") as datafile:
                s=datafile.read().split()
            d=s[0::3]
            x=np.array(s).reshape((-1,3))
            if Chprov in d and x.size >=3:
                arr= data_seg.spiketrains[i].as_array()
                where=d.index(Chprov)
                windowbeg=int(x[where][1])
                windowend=int(x[where][2])
                if windowend==-1:			
                    windowend=arr[-1]
                tosave= arr[np.where(np.logical_and(arr >= windowbeg , arr <= windowend) == True)]
                np.savetxt(Chprov+".txt", tosave) #Creates files with the Spiketimes.
            else:
                np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array())
    else:
        for i in range(n_spike_trains):
            Chprov = data_seg.spiketrains[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array()) #Creates files with the Spiketimes.
        
    if(n_spike_trains!=0):
       print(df)
       file = open("Channels_Label_LFP.txt", "w+")
       file.write(str(df))
       file.close()
	   
    #Create and Save Binary/.NPY files of Analog signals
    for j in range(n_analog_signals):
        #temp=data_seg.analogsignals[j].name.split(" ")[2][1:-1]
        bundle_len = len(data_seg.analogsignals[j].name.split(" "))
        if(bundle_len==1):
           np.save(data_seg.analogsignals[j].name.split(" ")[0], data_seg.analogsignals[j].as_array())
        else:
           list_channels_raw = data_seg.analogsignals[j].name.split(" ")[2][1:-1]
           list_channels=list_channels_raw.split(",")
           nb_channels=len(list_channels)
           channels=data_seg.analogsignals[j].as_array()
           for c in range(nb_channels):
               np.save(list_channels[c], channels[:,c])
    
    ##Create and Save Binary/.NPY files of Analog signals
    #for j in range(n_analog_signals):
    #    #temp=data_seg.analogsignals[j].name.split(" ")[2][1:-1] 
    #    temp=data_seg.analogsignals[j].name.split(" ")[0][:] 
    #    #temp=data_seg.analogsignals[j].name
    #    np.save(temp, data_seg.analogsignals[j].as_array())
    
    #Create and Save Summary about the File
    an=["File of origin: " + data.file_origin, "Number of AnalogSignals: " + str(n_analog_signals)]
    for k in range(n_analog_signals):
        anlenght= str(data.children_recur[k].size)
        anunit=str(data.children_recur[k].units).split(" ")[1]
        anname=str(data.children_recur[k].name)
        antime = str(str(data.children_recur[k].t_start) + " to " + str(data.children_recur[k].t_stop))
        an+=[["Analog index:" + str(k) + " Channel Name: " + anname, "Lenght: "+ anlenght, " Unit: " + anunit, " Sampling Rate: " + str(ansampling_rate) + " Duration: " + antime]]    
    
    spk=["Number of SpikeTrains: " + str(n_spike_trains)]    
    for l in range(n_analog_signals, n_spike_trains + n_analog_signals):
        spkid = str(data.children_recur[l].annotations["id"])     
        spkcreated = str(data.children_recur[l].annotations["comment"])
        spkname= str(data.children_recur[l].name)
        spksize = str(data.children_recur[l].size)
        spkunit = str(data.children_recur[l].units).split(" ")[1]
        spk+=[["SpikeTrain index:" + str(l-n_analog_signals) + " Channel Id: " + spkid, " " + spkcreated, " Name: " + spkname, " Size: "+ spksize, " Unit: " + spkunit]]
    final = an + spk
    with open("summary.txt", "w+") as f:
        for item in final:
            f.write("%s\n" % "".join(item))
    f.close()     
    os.chdir("..")
    print("\n"+"All files were created!")

## 
#
# This function will is similar to createsave but for recordings done with a tetrode
#  
#  
#  file is the .smr file of the recording
def createsave_tetr(file):
    data, data_seg= read(file)
    reader = neo.io.Spike2IO(filename=file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
   
    #Create new folder and change directory
    today= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(today)
    os.chdir(os.path.expanduser(today))
    
    #Create DataFrame (LFP should be indicated by the subject) and SpikeTime files
    res=[]
    LFP = input("Enter LFP number:")
    if os.path.isfile("..//unitswindow.txt"):
        for i in range(n_spike_trains):
            Chprov = data.list_units[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            with open("..//unitswindow.txt", "r") as datafile:
                s=datafile.read().split()
            d=s[0::3]
            x=np.array(s).reshape((-1,3))
            if Chprov in d and x.size >=3:
                arr= data_seg.spiketrains[i].as_array()
                where=d.index(Chprov)
                windowbeg=int(x[where][1])
                windowend=int(x[where][2])
                if windowend==-1:			
                    windowend=arr[-1]
                tosave= arr[np.where(np.logical_and(arr >= windowbeg , arr <= windowend) == True)]
                np.savetxt(Chprov+".txt", tosave) #Creates files with the Spiketimes.
            else:
                np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array())
    else:
        for i in range(n_spike_trains):
            Chprov = data.list_units[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array()) #Creates files with the Spiketimes.
        
    if(n_spike_trains!=0):
       print(df)
       file = open("Channels_Label_LFP.txt", "w+")
       file.write(str(df))
       file.close()
    
    #Create and Save Binary/.NPY files of Analog signals
    for j in range(n_analog_signals):
        #temp=data_seg.analogsignals[j].name.split(" ")[2][1:-1]
        bundle_len = len(data_seg.analogsignals[j].name.split(" "))
        if(bundle_len==1):
           np.save(data_seg.analogsignals[j].name.split(" ")[0], data_seg.analogsignals[j].as_array())
        else:
           list_channels_raw = data_seg.analogsignals[j].name.split(" ")[2][1:-1]
           list_channels=list_channels_raw.split(",")
           nb_channels=len(list_channels)
           channels=data_seg.analogsignals[j].as_array()
           for c in range(nb_channels):
               np.save(list_channels[c], channels[:,c])
    
    #Create and Save Summary about the File
    an=["File of origin: " + data.file_origin, "Number of AnalogSignals: " + str(n_analog_signals)]
    for k in range(n_analog_signals):
        anlenght= str(data.children_recur[k].size)
        anunit=str(data.children_recur[k].units).split(" ")[1]
        anname=str(data.children_recur[k].name)
        antime = str(str(data.children_recur[k].t_start) + " to " + str(data.children_recur[k].t_stop))
        an+=[["Analog index:" + str(k) + " Channel Name: " + anname, "Lenght: "+ anlenght, " Unit: " + anunit, " Sampling Rate: " + str(ansampling_rate) + " Duration: " + antime]]    
    
    spk=["Number of SpikeTrains: " + str(n_spike_trains)]    
    for l in range(n_analog_signals, n_spike_trains + n_analog_signals):
        spkid = str(data.children_recur[l].annotations["id"])     
        spkcreated = str(data.children_recur[l].annotations["comment"])
        spkname= str(data.children_recur[l].name)
        spksize = str(data.children_recur[l].size)
        spkunit = str(data.children_recur[l].units).split(" ")[1]
        spk+=[["SpikeTrain index:" + str(l-n_analog_signals) + " Channel Id: " + spkid, " " + spkcreated, " Name: " + spkname, " Size: "+ spksize, " Unit: " + spkunit]]
    final = an + spk
    with open("summary.txt", "w+") as f:
        for item in final:
            f.write("%s\n" % "".join(item))
    f.close()     
    os.chdir("..")
    print("\n"+"All files were created!")
		


		
## 
#
# Plots the spike shapes during baseline and during motifs. Raw and average spike shapes from filtered and unfiltered LFP
#
# Arguments:
#
# spikefile is the spike file
#
# raw is the .npy file containing the Raw unfiltered neuronal signal
#
# rawfiltered is the .npy containing the spike2 filtered neuronal signal  
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif
#
# basebeg is the time of the start of the baseline period
#
# besend is the time of the end of the baseline period
#  
def spikeshapes_relevant_mean_spikefile(spikefile, raw, rawfiltered,motifile,basebeg,basend,fs=fs):
    LFP=np.load(raw)
    notLFP=np.load(rawfiltered)
    windowsize=int(fs*4/1000) #Define here the number of points that suit your window (set to 2ms)
    x_tim = np.arange(windowsize)/(fs/1000)
    n_neurons_spk_shpe=50
    
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    if(len(noisy_motifs)!=0):
       all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)
    else:
       all_motifs=clean_motifs

    shoulder_beg= 0.02 #in seconds
    shoulder_end= 0.02 #in seconds	
	
    LFP_shift=105
    notLFP_shift=67
	
    nb_motifs=len(all_motifs[:,0]) #number of sung motifs (noisy or not)
	
	# Create and save the spikeshapes
    # This part will iterate through all the .txt files containing the spiketimes inside the folder.
    answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")
    # Spike shapes during the baseline
    channel1 = np.loadtxt(spikefile)
    print(spikefile)
    print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
    x1=np.empty([1,windowsize+2],int)
    x2=np.empty([1,windowsize+2],int)
    idx_baseline=np.where(np.logical_and(channel1 >= basebeg, channel1 <= basend) == True)
    idx_baseline=idx_baseline[0]
    for n in idx_baseline:
        a1= int(channel1[n]*fs)-LFP_shift
        if a1 < 0:
            continue
        analogtxt1=LFP[a1:a1+windowsize].reshape(1,windowsize)
        y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
        res1 = np.append(y1,analogtxt1).reshape(1,-1)
        x1=np.append(x1,res1, axis=0)
		
        a1=a1+notLFP_shift
        analogtxt2=notLFP[a1:a1+windowsize].reshape(1,windowsize)
        y2 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
        res2 = np.append(y2,analogtxt2).reshape(1,-1)
        x2=np.append(x2,res2, axis=0)                     		
		
    b1=x1[1:]   	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_LFP_baseline#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    b2=x2[1:]   	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_Filt_baseline#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    
    if ((answer4 == "") or (answer4.lower()[0] == "y") and (len(b1)>n_neurons_spk_shpe)):
       n_shapes=0
       plot_shapes=True
       while True:
           if(plot_shapes):
             py.fig,s = py.subplots(2,2)
             x = random.sample(range(0,len(b1)), n_neurons_spk_shpe)
             #Plot mean spike/LFP shape
             samples_LFP=[]
             samples_notLFP=[]
             spk_base=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             lfp_base=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             for i in x:
                 window1=int(b1[i][0])
                 window2=int(b1[i][1])
                 LFP_=LFP[window1:window2]-np.mean(LFP[window1:window2])
                 aux1 = np.array([[window1],[window2]], np.int32).reshape(1,2)
                 aux2 = np.append(aux1,LFP_).reshape(1,-1)
                 lfp_base=np.append(lfp_base,aux2, axis=0)
                 notLFP_=notLFP[window1+notLFP_shift:window2+notLFP_shift]-np.mean(notLFP[window1+notLFP_shift:window2+notLFP_shift])
                 aux1 = np.array([[window1],[window2]], np.int32).reshape(1,2)
                 aux2 = np.append(aux1,notLFP_).reshape(1,-1)
                 spk_base=np.append(spk_base,aux2, axis=0)                     
                 s[0,0].plot(x_tim,LFP_,color="black")
                 s[1,0].plot(x_tim,notLFP_,color="black")
                 samples_LFP+=[LFP_]
                 samples_notLFP+=[notLFP_]			   
	 
             samples_LFP=np.array(samples_LFP)
             samples_notLFP=np.array(samples_notLFP)
             s[0,1].plot(x_tim,np.mean(samples_LFP,axis=0),color="black")
             s[1,1].plot(x_tim,np.mean(samples_notLFP,axis=0),color="black")
             s[0,1].set_title("Mean SpikeShape LFP base",fontsize=18)
             s[1,1].set_title("Mean SpikeShape filtered base",fontsize=18) # Just like you would see in Spike2
             s[0,1].set_ylabel("Amplitude",fontsize=18)
             s[1,1].set_ylabel("Amplitude",fontsize=18)
             s[1,1].set_xlabel("Time(ms)",fontsize=18)
             #py.tight_layout()
             s[0,0].set_title("SpikeShape LFP base",fontsize=18)
             s[1,0].set_title("SpikeShape filtered base",fontsize=18) # Just like you would see in Spike2
             s[0,0].set_ylabel("Amplitude",fontsize=18)
             s[1,0].set_ylabel("Amplitude",fontsize=18)
             s[1,0].set_xlabel("Time(ms)",fontsize=18)
             #py.tight_layout()
	 
           py.waitforbuttonpress(1)
           print("Happy? Key click for yes, mouse click for no")
           while True:
               if py.waitforbuttonpress():
                   plot_shapes=False #key pressed: stop plotting spike shapes
                   break
               else:
                   break
	  
           if(plot_shapes==False):#key pressed: stop plotting spike shapes
             b1=spk_base[1:]
             np.savetxt("SpikeShape_baseline_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             b1=lfp_base[1:]
             np.savetxt("LFPShape_baseline_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             break


    #answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")		
    # Spike shapes during the motifs
    channel1 = np.loadtxt(spikefile)
    print(spikefile)
    print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
    x1=np.empty([1,windowsize+2],int)
    x2=np.empty([1,windowsize+2],int)
    spk_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
    lfp_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
    for i in range(nb_motifs):
        motif_on=(all_motifs[i,0]/fs)-shoulder_beg#onst of motif
        motif_off=(all_motifs[i,-1]/fs)+shoulder_end#offset of motif
        idx_baseline=np.where(np.logical_and(channel1 >= motif_on, channel1 <= motif_off) == True)
        idx_baseline=idx_baseline[0]
        for n in idx_baseline:
            a1= int(channel1[n]*fs)-LFP_shift
            if a1 < 0:
                continue
            analogtxt1=LFP[a1:a1+windowsize].reshape(1,windowsize)
            y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1 = np.append(y1,analogtxt1).reshape(1,-1)
            x1=np.append(x1,res1, axis=0)
		    
            a1=a1+notLFP_shift
            analogtxt2=notLFP[a1:a1+windowsize].reshape(1,windowsize)
            y2 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res2 = np.append(y2,analogtxt2).reshape(1,-1)
            x2=np.append(x2,res2, axis=0)                     		
		
    b1=x1[1:]   	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_LFP_motif#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    b2=x2[1:]   	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_Filt_motif#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    print(len(b1))
    if ((answer4 == "") or (answer4.lower()[0] == "y") and (len(b1)>n_neurons_spk_shpe)):
       n_shapes=0
       plot_shapes=True
       while True:
           if(plot_shapes):
             py.fig,s = py.subplots(2,2)
             x = random.sample(range(1,len(b1)), n_neurons_spk_shpe)
             #Plot mean spike/LFP shape
             samples_LFP=[]
             samples_notLFP=[]
             spk_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             lfp_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             for i in x:
                 window1=int(b1[i][0])
                 window2=int(b1[i][1])
                 LFP_=LFP[window1:window2]-np.mean(LFP[window1:window2])
                 aux1 = np.array([[window1],[window2]], np.int32).reshape(1,2)
                 aux2 = np.append(aux1,LFP_).reshape(1,-1)
                 lfp_motif=np.append(lfp_motif,aux2, axis=0)
                 notLFP_=notLFP[window1+notLFP_shift:window2+notLFP_shift]-np.mean(notLFP[window1+notLFP_shift:window2+notLFP_shift])
                 aux1 = np.array([[window1],[window2]], np.int32).reshape(1,2)
                 aux2 = np.append(aux1,notLFP_).reshape(1,-1)
                 spk_motif=np.append(spk_motif,aux2, axis=0)                     
                 s[0,0].plot(x_tim,LFP_,color="black")
                 s[1,0].plot(x_tim,notLFP_,color="black")
                 samples_LFP+=[LFP_]
                 samples_notLFP+=[notLFP_]	
	 
             samples_LFP=np.array(samples_LFP)
             samples_notLFP=np.array(samples_notLFP)
             s[0,1].plot(x_tim,np.mean(samples_LFP,axis=0),color="black")
             s[1,1].plot(x_tim,np.mean(samples_notLFP,axis=0),color="black")
             s[0,1].set_title("Mean SpikeShape LFP motif",fontsize=18)
             s[1,1].set_title("Mean SpikeShape filtered motif",fontsize=18) # Just like you would see in Spike2
             s[0,1].set_ylabel("Amplitude",fontsize=18)
             s[1,1].set_ylabel("Amplitude",fontsize=18)
             s[1,1].set_xlabel("Time(ms)",fontsize=18)
             #py.tight_layout()
             s[0,0].set_title("SpikeShape LFP motif",fontsize=18)
             s[1,0].set_title("SpikeShape filtered motif",fontsize=18) # Just like you would see in Spike2
             s[0,0].set_ylabel("Amplitude",fontsize=18)
             s[1,0].set_ylabel("Amplitude",fontsize=18)
             s[1,0].set_xlabel("Time(ms)",fontsize=18)
             #py.tight_layout()
	 
           py.waitforbuttonpress(1)
           print("Happy? Key click for yes, mouse click for no")
           while True:
               if py.waitforbuttonpress():
                   plot_shapes=False
                   break
               else:
                   break
	  
           if(plot_shapes==False):#key pressed: stop plotting spike shapes
             b1=spk_motif[1:]
             np.savetxt("SpikeShape_motif_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             b1=lfp_motif[1:]
             np.savetxt("LFPShape_motif_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             break
	

## 
#
# As in spikeshapes_relevant_mean_spikefile but for tetreode recordings 
#
# Arguments:
#
# spikefile is the spike file
#
# raw is the .npy file containing the Raw unfiltered neuronal signal
#
# rawfiltered is the .npy containing the spike2 filtered neuronal signal  
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif
#
# basebeg is the time of the start of the baseline period
#
# besend is the time of the end of the baseline period
#  
def spikeshapes_relevant_mean_spikefile_tetr(spikefile, raw1, raw2, raw3, raw4, rawfiltered1, rawfiltered2, rawfiltered3, rawfiltered4, motifile,basebeg,basend,fs=fs):
    LFP1=np.load(raw1)
    LFP2=np.load(raw2)
    LFP3=np.load(raw3)
    LFP4=np.load(raw4)
	
    notLFP1=np.load(rawfiltered1)
    notLFP2=np.load(rawfiltered2)
    notLFP3=np.load(rawfiltered3)
    notLFP4=np.load(rawfiltered4)
	
    windowsize=int(fs*4/1000) #Define here the number of points that suit your window (set to 2ms)
    x_tim = np.arange(windowsize)/(fs/1000)
    n_neurons_spk_shpe=50
    
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    if(len(noisy_motifs)!=0):
       all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)
    else:
       all_motifs=clean_motifs

    shoulder_beg= 0.02 #in seconds
    shoulder_end= 0.02 #in seconds	
	
    LFP_shift=125
    notLFP_shift=67
	
    nb_motifs=len(all_motifs[:,0]) #number of sung motifs (noisy or not)
	
	# Create and save the spikeshapes
    # This part will iterate through all the .txt files containing the spiketimes inside the folder.
    answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")
	#######################################################
    #      Spike shapes during the baseline
	#######################################################
    channel1 = np.loadtxt(spikefile)
    print(spikefile)
    print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
    x1=np.empty([1,windowsize+2],int)
    x2=np.empty([1,windowsize+2],int)
    x3=np.empty([1,windowsize+2],int)
    x4=np.empty([1,windowsize+2],int)
    x1n=np.empty([1,windowsize+2],int)
    x2n=np.empty([1,windowsize+2],int)
    x3n=np.empty([1,windowsize+2],int)
    x4n=np.empty([1,windowsize+2],int)
	
    idx_baseline=np.where(np.logical_and(channel1 >= basebeg, channel1 <= basend) == True)
    idx_baseline=idx_baseline[0]
    for n in idx_baseline:
        a1= int(channel1[n]*fs)-LFP_shift
        if a1 < 0:
            continue
        analogtxt1=LFP1[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt2=LFP2[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt3=LFP3[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt4=LFP4[a1:a1+windowsize].reshape(1,windowsize)
		
        y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
        res1 = np.append(y1,analogtxt1).reshape(1,-1)
        res2 = np.append(y1,analogtxt2).reshape(1,-1)
        res3 = np.append(y1,analogtxt3).reshape(1,-1)
        res4 = np.append(y1,analogtxt4).reshape(1,-1)
        x1=np.append(x1,res1, axis=0)
        x2=np.append(x2,res2, axis=0)
        x3=np.append(x3,res3, axis=0)
        x4=np.append(x4,res4, axis=0)
		
        a1=a1+notLFP_shift
        analogtxt1n=notLFP1[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt2n=notLFP2[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt3n=notLFP3[a1:a1+windowsize].reshape(1,windowsize)
        analogtxt4n=notLFP4[a1:a1+windowsize].reshape(1,windowsize)
		
        y2 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
        res1n = np.append(y2,analogtxt1n).reshape(1,-1)
        res2n = np.append(y2,analogtxt2n).reshape(1,-1)
        res3n = np.append(y2,analogtxt3n).reshape(1,-1)
        res4n = np.append(y2,analogtxt4n).reshape(1,-1)
        x1n=np.append(x1n,res1n, axis=0)    
        x2n=np.append(x2n,res2n, axis=0) 
        x3n=np.append(x3n,res3n, axis=0) 
        x4n=np.append(x4n,res4n, axis=0) 		
		
    b1=x1[1:]   
    b2=x2[1:]
    b3=x3[1:]
    b4=x4[1:]	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_LFP1_baseline#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP2_baseline#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP3_baseline#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP4_baseline#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")

    b1n=x1n[1:]   
    b2n=x2n[1:]
    b3n=x3n[1:]
    b4n=x4n[1:]  	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_Filt1_baseline#"+spikefile, b1n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt2_baseline#"+spikefile, b2n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt3_baseline#"+spikefile, b3n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt4_baseline#"+spikefile, b4n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    
	
	#Plot spike shapes
    if ((answer4 == "") or (answer4.lower()[0] == "y") and (len(b1)>n_neurons_spk_shpe)):
       mx_LFP=0
       mn_LFP=0
       mx_notLFP=0
       mn_notLFP=0
       mx_LFP_mean=0
       mn_LFP_mean=0
       mx_notLFP_mean=0
       mn_notLFP_mean=0
       n_shapes=0
       plot_shapes=True
       while True:
           if(plot_shapes):
             py.fig,s_LFP = py.subplots(2,2)
             py.fig,s_notLFP = py.subplots(2,2)
             py.fig,s_LFP_mean = py.subplots(2,2)
             py.fig,s_notLFP_mean = py.subplots(2,2)
             x = random.sample(range(0,len(b1)), n_neurons_spk_shpe)
             #Plot mean spike/LFP shape
             samples_LFP1=[]
             samples_LFP2=[]
             samples_LFP3=[]
             samples_LFP4=[]
			 
             samples_notLFP1=[]
             samples_notLFP2=[]
             samples_notLFP3=[]
             samples_notLFP4=[]
	
             spk1_base=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk2_base=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk3_base=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk4_base=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             lfp1_base=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp2_base=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp3_base=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp4_base=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             for i in x:
                 window1_1=int(b1[i][0])
                 window1_2=int(b1[i][1])
                 window2_1=int(b2[i][0])
                 window2_2=int(b2[i][1])
                 window3_1=int(b3[i][0])
                 window3_2=int(b3[i][1])
                 window4_1=int(b4[i][0])
                 window4_2=int(b4[i][1])
                 LFP1_=LFP1[window1_1:window1_2]-np.mean(LFP1[window1_1:window1_2])
                 LFP2_=LFP2[window2_1:window2_2]-np.mean(LFP2[window2_1:window2_2])
                 LFP3_=LFP3[window3_1:window3_2]-np.mean(LFP3[window3_1:window3_2])
                 LFP4_=LFP4[window4_1:window4_2]-np.mean(LFP4[window4_1:window4_2])
				 
                 aux1_1 = np.array([[window1_1],[window1_2]], np.int32).reshape(1,2)
                 aux2_1 = np.array([[window2_1],[window2_2]], np.int32).reshape(1,2)
                 aux3_1 = np.array([[window3_1],[window3_2]], np.int32).reshape(1,2)
                 aux4_1 = np.array([[window4_1],[window4_2]], np.int32).reshape(1,2)
                 aux1_2 = np.append(aux1_1,LFP1_).reshape(1,-1)
                 aux2_2 = np.append(aux2_1,LFP2_).reshape(1,-1)
                 aux3_2 = np.append(aux3_1,LFP3_).reshape(1,-1)
                 aux4_2 = np.append(aux4_1,LFP4_).reshape(1,-1)
                 lfp1_base=np.append(lfp1_base,aux1_2, axis=0)
                 lfp2_base=np.append(lfp2_base,aux2_2, axis=0)
                 lfp3_base=np.append(lfp3_base,aux3_2, axis=0)
                 lfp4_base=np.append(lfp4_base,aux4_2, axis=0)
				 
                 notLFP1_=notLFP1[window1_1+notLFP_shift:window1_2+notLFP_shift]-np.mean(notLFP1[window1_1+notLFP_shift:window1_2+notLFP_shift])
                 notLFP2_=notLFP2[window2_1+notLFP_shift:window2_2+notLFP_shift]-np.mean(notLFP2[window2_1+notLFP_shift:window2_2+notLFP_shift])
                 notLFP3_=notLFP3[window3_1+notLFP_shift:window3_2+notLFP_shift]-np.mean(notLFP3[window3_1+notLFP_shift:window3_2+notLFP_shift])
                 notLFP4_=notLFP4[window4_1+notLFP_shift:window4_2+notLFP_shift]-np.mean(notLFP4[window4_1+notLFP_shift:window4_2+notLFP_shift])
                 aux1_1 = np.array([[window1_1],[window1_2]], np.int32).reshape(1,2)
                 aux2_1 = np.array([[window2_1],[window2_2]], np.int32).reshape(1,2)
                 aux3_1 = np.array([[window3_1],[window3_2]], np.int32).reshape(1,2)
                 aux4_1 = np.array([[window4_1],[window4_2]], np.int32).reshape(1,2)
                 aux1_2 = np.append(aux1_1,notLFP1_).reshape(1,-1)
                 aux2_2 = np.append(aux2_1,notLFP2_).reshape(1,-1)
                 aux3_2 = np.append(aux3_1,notLFP3_).reshape(1,-1)
                 aux4_2 = np.append(aux4_1,notLFP4_).reshape(1,-1)
                 spk1_base=np.append(spk1_base,aux1_2, axis=0)
                 spk2_base=np.append(spk2_base,aux2_2, axis=0)
                 spk3_base=np.append(spk3_base,aux3_2, axis=0)
                 spk4_base=np.append(spk4_base,aux4_2, axis=0)
				 				
                 mx_LFP=np.max([np.max(LFP1_),np.max(LFP2_),np.max(LFP3_),np.max(LFP4_),mx_LFP])
                 mn_LFP=np.min([np.min(LFP1_),np.min(LFP2_),np.min(LFP3_),np.min(LFP4_),mn_LFP])
                 s_LFP[0,0].plot(x_tim,LFP1_,color="black")
                 s_LFP[0,1].plot(x_tim,LFP2_,color="black")
                 s_LFP[1,0].plot(x_tim,LFP3_,color="black")
                 s_LFP[1,1].plot(x_tim,LFP4_,color="black")

                 mx_notLFP=np.max([np.max(notLFP1_),np.max(notLFP2_),np.max(notLFP3_),np.max(notLFP4_),mx_notLFP])
                 mn_notLFP=np.min([np.min(notLFP1_),np.min(notLFP2_),np.min(notLFP3_),np.min(notLFP4_),mn_notLFP])
                 s_notLFP[0,0].plot(x_tim,notLFP1_,color="black")
                 s_notLFP[0,1].plot(x_tim,notLFP2_,color="black")
                 s_notLFP[1,0].plot(x_tim,notLFP3_,color="black")
                 s_notLFP[1,1].plot(x_tim,notLFP4_,color="black")

                 samples_LFP1+=[LFP1_]
                 samples_LFP2+=[LFP2_]
                 samples_LFP3+=[LFP3_]
                 samples_LFP4+=[LFP4_]
                 samples_notLFP1+=[notLFP1_]
                 samples_notLFP2+=[notLFP2_]	
                 samples_notLFP3+=[notLFP3_]	
                 samples_notLFP4+=[notLFP4_]					 
	 
	 
	 
             s_LFP[0,0].set_title("SpikeShape LFP1 base",fontsize=18)
             s_LFP[0,1].set_title("SpikeShape LFP2 base",fontsize=18)
             s_LFP[1,0].set_title("SpikeShape LFP3 base",fontsize=18)
             s_LFP[1,1].set_title("SpikeShape LFP4 base",fontsize=18)
             s_LFP[0,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP[0,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP[1,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP[1,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[1,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[0,0].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[0,1].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[1,0].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[1,1].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
			 
             s_notLFP[0,0].set_title("SpikeShape filt LFP1 base",fontsize=18)
             s_notLFP[0,1].set_title("SpikeShape filt LFP2 base",fontsize=18)
             s_notLFP[1,0].set_title("SpikeShape filt LFP3 base",fontsize=18)
             s_notLFP[1,1].set_title("SpikeShape filt LFP4 base",fontsize=18)
             s_notLFP[0,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[0,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[1,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[1,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[1,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[0,0].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[0,1].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[1,0].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[1,1].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
			 
             samples_LFP1=np.array(samples_LFP1)
             samples_LFP2=np.array(samples_LFP2)
             samples_LFP3=np.array(samples_LFP3)
             samples_LFP4=np.array(samples_LFP4)
             samples_notLFP1=np.array(samples_notLFP1)
             samples_notLFP2=np.array(samples_notLFP2)
             samples_notLFP3=np.array(samples_notLFP3)
             samples_notLFP4=np.array(samples_notLFP4)
			 
			 
             s_LFP_mean[0,0].plot(x_tim,np.mean(samples_LFP1,axis=0),color="black")
             s_LFP_mean[0,1].plot(x_tim,np.mean(samples_LFP2,axis=0),color="black")
             s_LFP_mean[1,0].plot(x_tim,np.mean(samples_LFP3,axis=0),color="black")
             s_LFP_mean[1,1].plot(x_tim,np.mean(samples_LFP4,axis=0),color="black")
             s_LFP_mean[0,0].set_title("Mean SpikeShape LFP1 base",fontsize=18)
             s_LFP_mean[0,1].set_title("Mean SpikeShape LFP2 base",fontsize=18)
             s_LFP_mean[1,0].set_title("Mean SpikeShape LFP3 base",fontsize=18)
             s_LFP_mean[1,1].set_title("Mean SpikeShape LFP4 base",fontsize=18)
             s_LFP_mean[0,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[0,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[1,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[1,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[1,1].set_xlabel("Time(ms)",fontsize=18)
             mx_LFP_mean=1.1*np.max([np.mean(samples_LFP1,axis=0),np.mean(samples_LFP2,axis=0),np.mean(samples_LFP3,axis=0),np.mean(samples_LFP4,axis=0)])
             mn_LFP_mean=1.1*np.min([np.mean(samples_LFP1,axis=0),np.mean(samples_LFP2,axis=0),np.mean(samples_LFP3,axis=0),np.mean(samples_LFP4,axis=0)])
             s_LFP_mean[0,0].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[0,1].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[1,0].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[1,1].set_ylim([mn_LFP_mean,mx_LFP_mean])
			 
             s_notLFP_mean[0,0].plot(x_tim,np.mean(samples_notLFP1,axis=0),color="black")
             s_notLFP_mean[0,1].plot(x_tim,np.mean(samples_notLFP2,axis=0),color="black")
             s_notLFP_mean[1,0].plot(x_tim,np.mean(samples_notLFP3,axis=0),color="black")
             s_notLFP_mean[1,1].plot(x_tim,np.mean(samples_notLFP4,axis=0),color="black")
             s_notLFP_mean[0,0].set_title("Mean SpikeShape filt LFP1 base",fontsize=18)
             s_notLFP_mean[0,1].set_title("Mean SpikeShape filt LFP2 base",fontsize=18)
             s_notLFP_mean[1,0].set_title("Mean SpikeShape filt LFP3 base",fontsize=18)
             s_notLFP_mean[1,1].set_title("Mean SpikeShape filt LFP4 base",fontsize=18)
             s_notLFP_mean[0,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[0,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[1,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[1,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[1,1].set_xlabel("Time(ms)",fontsize=18)
             mx_notLFP_mean=1.1*np.max([np.mean(samples_notLFP1,axis=0),np.mean(samples_notLFP2,axis=0),np.mean(samples_notLFP3,axis=0),np.mean(samples_notLFP4,axis=0)])
             mn_notLFP_mean=1.1*np.min([np.mean(samples_notLFP1,axis=0),np.mean(samples_notLFP2,axis=0),np.mean(samples_notLFP3,axis=0),np.mean(samples_notLFP4,axis=0)])
             s_notLFP_mean[0,0].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[0,1].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[1,0].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[1,1].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
			 #py.tight_layout()
			 
	 
           py.waitforbuttonpress(1)
           print("Happy? Key click for yes, mouse click for no")
           while True:
               if py.waitforbuttonpress():
                   plot_shapes=False #key pressed: stop plotting spike shapes
                   break
               else:
                   break
	  
           if(plot_shapes==False):#key pressed: stop plotting spike shapes
              b1=spk1_base[1:]
              b2=spk2_base[1:]
              b3=spk3_base[1:]
              b4=spk4_base[1:]
              np.savetxt("SpikeShape1_baseline_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
              np.savetxt("SpikeShape2_baseline_plot#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
              np.savetxt("SpikeShape3_baseline_plot#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
              np.savetxt("SpikeShape4_baseline_plot#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
              b1=lfp1_base[1:]
              b2=lfp2_base[1:]
              b3=lfp3_base[1:]
              b4=lfp4_base[1:]
              np.savetxt("LFPShape1_baseline_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
              np.savetxt("LFPShape2_baseline_plot#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
              np.savetxt("LFPShape3_baseline_plot#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
              np.savetxt("LFPShape4_baseline_plot#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
              break


    #answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")	
    ############################################################	
    #          Spike shapes during the motifs
	############################################################
    channel1 = np.loadtxt(spikefile)
    print(spikefile)
    print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
    x1=np.empty([1,windowsize+2],int)
    x2=np.empty([1,windowsize+2],int)
    x3=np.empty([1,windowsize+2],int)
    x4=np.empty([1,windowsize+2],int)
    x1n=np.empty([1,windowsize+2],int)
    x2n=np.empty([1,windowsize+2],int)
    x3n=np.empty([1,windowsize+2],int)
    x4n=np.empty([1,windowsize+2],int)
	
    for i in range(nb_motifs):
        motif_on=(all_motifs[i,0]/fs)-shoulder_beg#onst of motif
        motif_off=(all_motifs[i,-1]/fs)+shoulder_end#offset of motif

        idx_baseline=np.where(np.logical_and(channel1 >= motif_on, channel1 <= motif_off) == True)
        idx_baseline=idx_baseline[0]
        for n in idx_baseline:
            a1= int(channel1[n]*fs)-LFP_shift
            if a1 < 0:
                continue
				
            analogtxt1=LFP1[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt2=LFP2[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt3=LFP3[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt4=LFP4[a1:a1+windowsize].reshape(1,windowsize)
		    
            y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1 = np.append(y1,analogtxt1).reshape(1,-1)
            res2 = np.append(y1,analogtxt2).reshape(1,-1)
            res3 = np.append(y1,analogtxt3).reshape(1,-1)
            res4 = np.append(y1,analogtxt4).reshape(1,-1)
            x1=np.append(x1,res1, axis=0)
            x2=np.append(x2,res2, axis=0)
            x3=np.append(x3,res3, axis=0)
            x4=np.append(x4,res4, axis=0)
		    
            a1=a1+notLFP_shift
            analogtxt1n=notLFP1[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt2n=notLFP2[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt3n=notLFP3[a1:a1+windowsize].reshape(1,windowsize)
            analogtxt4n=notLFP4[a1:a1+windowsize].reshape(1,windowsize)
		    
            y2 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1n = np.append(y2,analogtxt1n).reshape(1,-1)
            res2n = np.append(y2,analogtxt2n).reshape(1,-1)
            res3n = np.append(y2,analogtxt3n).reshape(1,-1)
            res4n = np.append(y2,analogtxt4n).reshape(1,-1)
            x1n=np.append(x1n,res1n, axis=0)    
            x2n=np.append(x2n,res2n, axis=0) 
            x3n=np.append(x3n,res3n, axis=0) 
            x4n=np.append(x4n,res4n, axis=0) 		

    b1=x1[1:]   
    b2=x2[1:]
    b3=x3[1:]
    b4=x4[1:]	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_LFP1_motif#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP2_motif#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP3_motif#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_LFP4_motif#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")

    b1n=x1n[1:]   
    b2n=x2n[1:]
    b3n=x3n[1:]
    b4n=x4n[1:]  	
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape_Filt1_motif#"+spikefile, b1n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt2_motif#"+spikefile, b2n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt3_motif#"+spikefile, b3n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    np.savetxt("SpikeShape_Filt4_motif#"+spikefile, b4n, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
	
    if ((answer4 == "") or (answer4.lower()[0] == "y") and (len(b1)>n_neurons_spk_shpe)):
       mx_LFP=0
       mn_LFP=0
       mx_notLFP=0
       mn_notLFP=0
       mx_LFP_mean=0
       mn_LFP_mean=0
       mx_notLFP_mean=0
       mn_notLFP_mean=0
       n_shapes=0
       plot_shapes=True
       while True:
           if(plot_shapes):
             py.fig,s_LFP = py.subplots(2,2)
             py.fig,s_notLFP = py.subplots(2,2)
             py.fig,s_LFP_mean = py.subplots(2,2)
             py.fig,s_notLFP_mean = py.subplots(2,2)
             x = random.sample(range(0,len(b1)), n_neurons_spk_shpe)

             #Plot mean spike/LFP shape
             samples_LFP1=[]
             samples_LFP2=[]
             samples_LFP3=[]
             samples_LFP4=[]
			 
             samples_notLFP1=[]
             samples_notLFP2=[]
             samples_notLFP3=[]
             samples_notLFP4=[]

             spk1_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk2_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk3_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             spk4_motif=np.empty([1,windowsize+2],int)#contains the spikeshapes that will be saved
             lfp1_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp2_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp3_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved
             lfp4_motif=np.empty([1,windowsize+2],int)#contains the lfpshapes that will be saved

             for i in x:	 
                 window1_1=int(b1[i][0])
                 window1_2=int(b1[i][1])
                 window2_1=int(b2[i][0])
                 window2_2=int(b2[i][1])
                 window3_1=int(b3[i][0])
                 window3_2=int(b3[i][1])
                 window4_1=int(b4[i][0])
                 window4_2=int(b4[i][1])
                 LFP1_=LFP1[window1_1:window1_2]-np.mean(LFP1[window1_1:window1_2])
                 LFP2_=LFP2[window2_1:window2_2]-np.mean(LFP2[window2_1:window2_2])
                 LFP3_=LFP3[window3_1:window3_2]-np.mean(LFP3[window3_1:window3_2])
                 LFP4_=LFP4[window4_1:window4_2]-np.mean(LFP4[window4_1:window4_2])
				 
                 aux1_1 = np.array([[window1_1],[window1_2]], np.int32).reshape(1,2)
                 aux2_1 = np.array([[window2_1],[window2_2]], np.int32).reshape(1,2)
                 aux3_1 = np.array([[window3_1],[window3_2]], np.int32).reshape(1,2)
                 aux4_1 = np.array([[window4_1],[window4_2]], np.int32).reshape(1,2)
                 aux1_2 = np.append(aux1_1,LFP1_).reshape(1,-1)
                 aux2_2 = np.append(aux2_1,LFP2_).reshape(1,-1)
                 aux3_2 = np.append(aux3_1,LFP3_).reshape(1,-1)
                 aux4_2 = np.append(aux4_1,LFP4_).reshape(1,-1)
                 lfp1_motif=np.append(lfp1_motif,aux1_2, axis=0)
                 lfp2_motif=np.append(lfp2_motif,aux2_2, axis=0)
                 lfp3_motif=np.append(lfp3_motif,aux3_2, axis=0)
                 lfp4_motif=np.append(lfp4_motif,aux4_2, axis=0)
				 
                 notLFP1_=notLFP1[window1_1+notLFP_shift:window1_2+notLFP_shift]-np.mean(notLFP1[window1_1+notLFP_shift:window1_2+notLFP_shift])
                 notLFP2_=notLFP2[window2_1+notLFP_shift:window2_2+notLFP_shift]-np.mean(notLFP2[window2_1+notLFP_shift:window2_2+notLFP_shift])
                 notLFP3_=notLFP3[window3_1+notLFP_shift:window3_2+notLFP_shift]-np.mean(notLFP3[window3_1+notLFP_shift:window3_2+notLFP_shift])
                 notLFP4_=notLFP4[window4_1+notLFP_shift:window4_2+notLFP_shift]-np.mean(notLFP4[window4_1+notLFP_shift:window4_2+notLFP_shift])
                 aux1_1 = np.array([[window1_1],[window1_2]], np.int32).reshape(1,2)
                 aux2_1 = np.array([[window2_1],[window2_2]], np.int32).reshape(1,2)
                 aux3_1 = np.array([[window3_1],[window3_2]], np.int32).reshape(1,2)
                 aux4_1 = np.array([[window4_1],[window4_2]], np.int32).reshape(1,2)
                 aux1_2 = np.append(aux1_1,notLFP1_).reshape(1,-1)
                 aux2_2 = np.append(aux2_1,notLFP2_).reshape(1,-1)
                 aux3_2 = np.append(aux3_1,notLFP3_).reshape(1,-1)
                 aux4_2 = np.append(aux4_1,notLFP4_).reshape(1,-1)
                 spk1_motif=np.append(spk1_motif,aux1_2, axis=0)
                 spk2_motif=np.append(spk2_motif,aux2_2, axis=0)
                 spk3_motif=np.append(spk3_motif,aux3_2, axis=0)
                 spk4_motif=np.append(spk4_motif,aux4_2, axis=0)
				 				
                 mx_LFP=np.max([np.max(LFP1_),np.max(LFP2_),np.max(LFP3_),np.max(LFP4_),mx_LFP])
                 mn_LFP=np.min([np.min(LFP1_),np.min(LFP2_),np.min(LFP3_),np.min(LFP4_),mn_LFP])
                 s_LFP[0,0].plot(x_tim,LFP1_,color="black")
                 s_LFP[0,1].plot(x_tim,LFP2_,color="black")
                 s_LFP[1,0].plot(x_tim,LFP3_,color="black")
                 s_LFP[1,1].plot(x_tim,LFP4_,color="black")

                 mx_notLFP=np.max([np.max(notLFP1_),np.max(notLFP2_),np.max(notLFP3_),np.max(notLFP4_),mx_notLFP])
                 mn_notLFP=np.min([np.min(notLFP1_),np.min(notLFP2_),np.min(notLFP3_),np.min(notLFP4_),mn_notLFP])
                 s_notLFP[0,0].plot(x_tim,notLFP1_,color="black")
                 s_notLFP[0,1].plot(x_tim,notLFP2_,color="black")
                 s_notLFP[1,0].plot(x_tim,notLFP3_,color="black")
                 s_notLFP[1,1].plot(x_tim,notLFP4_,color="black")

                 samples_LFP1+=[LFP1_]
                 samples_LFP2+=[LFP2_]
                 samples_LFP3+=[LFP3_]
                 samples_LFP4+=[LFP4_]
                 samples_notLFP1+=[notLFP1_]
                 samples_notLFP2+=[notLFP2_]	
                 samples_notLFP3+=[notLFP3_]	
                 samples_notLFP4+=[notLFP4_]	
	 
             
             s_LFP[0,0].set_title("SpikeShape LFP1 motif",fontsize=18)
             s_LFP[0,1].set_title("SpikeShape LFP2 motif",fontsize=18)
             s_LFP[1,0].set_title("SpikeShape LFP3 motif",fontsize=18)
             s_LFP[1,1].set_title("SpikeShape LFP4 motif",fontsize=18)
             s_LFP[0,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP[0,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP[1,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP[1,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[1,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP[0,0].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[0,1].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[1,0].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
             s_LFP[1,1].set_ylim([1.1*mn_LFP,1.1*mx_LFP])
			 
             s_notLFP[0,0].set_title("SpikeShape filt LFP1 motif",fontsize=18)
             s_notLFP[0,1].set_title("SpikeShape filt LFP2 motif",fontsize=18)
             s_notLFP[1,0].set_title("SpikeShape filt LFP3 motif",fontsize=18)
             s_notLFP[1,1].set_title("SpikeShape filt LFP4 motif",fontsize=18)
             s_notLFP[0,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[0,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[1,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[1,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[1,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP[0,0].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[0,1].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[1,0].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
             s_notLFP[1,1].set_ylim([1.1*mn_notLFP,1.1*mx_notLFP])
			 
             samples_LFP1=np.array(samples_LFP1)
             samples_LFP2=np.array(samples_LFP2)
             samples_LFP3=np.array(samples_LFP3)
             samples_LFP4=np.array(samples_LFP4)
             samples_notLFP1=np.array(samples_notLFP1)
             samples_notLFP2=np.array(samples_notLFP2)
             samples_notLFP3=np.array(samples_notLFP3)
             samples_notLFP4=np.array(samples_notLFP4)
			 
             s_LFP_mean[0,0].plot(x_tim,np.mean(samples_LFP1,axis=0),color="black")
             s_LFP_mean[0,1].plot(x_tim,np.mean(samples_LFP2,axis=0),color="black")
             s_LFP_mean[1,0].plot(x_tim,np.mean(samples_LFP3,axis=0),color="black")
             s_LFP_mean[1,1].plot(x_tim,np.mean(samples_LFP4,axis=0),color="black")
             s_LFP_mean[0,0].set_title("Mean SpikeShape LFP1 motif",fontsize=18)
             s_LFP_mean[0,1].set_title("Mean SpikeShape LFP2 motif",fontsize=18)
             s_LFP_mean[1,0].set_title("Mean SpikeShape LFP3 motif",fontsize=18)
             s_LFP_mean[1,1].set_title("Mean SpikeShape LFP4 motif",fontsize=18)
             s_LFP_mean[0,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[0,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[1,0].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[1,1].set_ylabel("Amplitude",fontsize=18)
             s_LFP_mean[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_LFP_mean[1,1].set_xlabel("Time(ms)",fontsize=18)
             mx_LFP_mean=1.1*np.max([np.mean(samples_LFP1,axis=0),np.mean(samples_LFP2,axis=0),np.mean(samples_LFP3,axis=0),np.mean(samples_LFP4,axis=0)])
             mn_LFP_mean=1.1*np.min([np.mean(samples_LFP1,axis=0),np.mean(samples_LFP2,axis=0),np.mean(samples_LFP3,axis=0),np.mean(samples_LFP4,axis=0)])
             s_LFP_mean[0,0].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[0,1].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[1,0].set_ylim([mn_LFP_mean,mx_LFP_mean])
             s_LFP_mean[1,1].set_ylim([mn_LFP_mean,mx_LFP_mean])
			 
             s_notLFP_mean[0,0].plot(x_tim,np.mean(samples_notLFP1,axis=0),color="black")
             s_notLFP_mean[0,1].plot(x_tim,np.mean(samples_notLFP2,axis=0),color="black")
             s_notLFP_mean[1,0].plot(x_tim,np.mean(samples_notLFP3,axis=0),color="black")
             s_notLFP_mean[1,1].plot(x_tim,np.mean(samples_notLFP4,axis=0),color="black")
             s_notLFP_mean[0,0].set_title("Mean SpikeShape filt LFP1 motif",fontsize=18)
             s_notLFP_mean[0,1].set_title("Mean SpikeShape filt LFP2 motif",fontsize=18)
             s_notLFP_mean[1,0].set_title("Mean SpikeShape filt LFP3 motif",fontsize=18)
             s_notLFP_mean[1,1].set_title("Mean SpikeShape filt LFP4 motif",fontsize=18)
             s_notLFP_mean[0,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[0,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[1,0].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[1,1].set_ylabel("Amplitude",fontsize=18)
             s_notLFP_mean[0,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[0,1].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[1,0].set_xlabel("Time(ms)",fontsize=18)
             s_notLFP_mean[1,1].set_xlabel("Time(ms)",fontsize=18)
             mx_notLFP_mean=1.1*np.max([np.mean(samples_notLFP1,axis=0),np.mean(samples_notLFP2,axis=0),np.mean(samples_notLFP3,axis=0),np.mean(samples_notLFP4,axis=0)])
             mn_notLFP_mean=1.1*np.min([np.mean(samples_notLFP1,axis=0),np.mean(samples_notLFP2,axis=0),np.mean(samples_notLFP3,axis=0),np.mean(samples_notLFP4,axis=0)])
             s_notLFP_mean[0,0].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[0,1].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[1,0].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
             s_notLFP_mean[1,1].set_ylim([mn_notLFP_mean,mx_notLFP_mean])
			 #py.tight_layout()

           py.waitforbuttonpress(1)
           print("Happy? Key click for yes, mouse click for no")
           while True:
               if py.waitforbuttonpress():
                   plot_shapes=False
                   break
               else:
                   break
	  
           if(plot_shapes==False):#key pressed: stop plotting spike shapes
             b1=spk1_motif[1:]
             b2=spk2_motif[1:]
             b3=spk3_motif[1:]
             b4=spk4_motif[1:]
             np.savetxt("SpikeShape1_motif_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             np.savetxt("SpikeShape2_motif_plot#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             np.savetxt("SpikeShape3_motif_plot#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             np.savetxt("SpikeShape4_motif_plot#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
             b1=lfp1_motif[1:]
             b2=lfp2_motif[1:]
             b3=lfp3_motif[1:]
             b4=lfp4_motif[1:]
             np.savetxt("LFPShape1_motif_plot#"+spikefile, b1, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             np.savetxt("LFPShape2_motif_plot#"+spikefile, b2, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             np.savetxt("LFPShape3_motif_plot#"+spikefile, b3, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             np.savetxt("LFPShape4_motif_plot#"+spikefile, b4, header="First column = Initial Time; Second column = Final Time; Third Column = First LFP Shape value, etc")
             break

## 
#
# This function will downsample your LFP signal to 1000Hz and save it as .npy file
def lfpdown(LFPfile, fs=fs): #LFPfile is the .npy one inside the new folder generated by the function createsave (for example, CSC1.npy)
    fs1=int(fs/1000)
    rawsignal=np.load(LFPfile)
    def mma(series,window):
        return np.convolve(series,np.repeat(1,window)/window,"same")
    
    rawsignal=rawsignal[0:][:,0] #window of the array, in case you want to select a specific part
    conv=mma(rawsignal,100) #convolved version
    c=[]
    for i in range(len(conv)):
        if i%fs1==0:
            c+=[conv[i]]
            
    downsamp=np.array(c)
    np.save("LFPDownsampled", downsamp)
    answer=input("Want to see the plots? Might be a bit heavy. [Y/n]")
    if answer == "" or answer.lower()[0] == "y":
        py.fig,(s,s1) = py.subplots(2,1) 
        s.plot(rawsignal)
        s.plot(conv)
        s.set_title("Plot of RawSignal X Convolved Version")
        s1.plot(downsamp)
        s1.set_title("LFP Downsampled")
        s.set_ylabel("Amplitude")
        s1.set_ylabel("Amplitude")
        s1.set_xlabel("Sample points")
        py.show()
        py.tight_layout()


## 
#
# This function generates spectrogram of the motifs in the song raw signal. 
# To be used with the new matfiles.
#    
# Arguments:
#    
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# fs = sampling frequency (Hz)
def spectrogram(songfile, beg, end, fs=fs):
    analog= np.load(songfile)
    rawsong1=analog[beg:end].reshape(1,-1)
    rawsong=rawsong1[0]
    #Compute and plot spectrogram
    #(f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, scaling="density", mode="complex")
    py.fig, ax = py.subplots(2,1)
    ax[0].plot(rawsong)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Sample points")
    _,_,_,im = ax[1].specgram(rawsong,Fs=fs, NFFT=980, noverlap=930, scale_by_freq=False, mode="default", pad_to=915, cmap="inferno")
    #py.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none", cmap="inferno")
    ax[1].tick_params(
                        axis="x",          # changes apply to the x-axis
                        which="both",      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
    ax[1].set_ylabel("Frequency")
    cbar=py.colorbar(im, ax=ax[1])
    cbar.ax.invert_yaxis()
    cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5, dtype=float))
    cbar.ax.set_yticklabels(np.floor(np.linspace(np.floor(cbar.vmin), cbar.vmax, 5)).astype(int))
    py.tight_layout() 


## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles.
# Based on psth_old but sets different figure size (25,12) instead of (18,15). No interpolation in the psth
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    finallist=sortsyls(motifile,0)
    #print(finallist)
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 0.05 #50 ms
    meandurall=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,len(finallist), figsize=(25,12), sharey=False)
    #fig, ax = plt.subplots(2,len(finallist), sharey=False,figsize=(25,12))
    for i in range(len(finallist)):
        if len(finallist) == 1:
            shapes = (1,)
            shapes2 = (0,)
        else:
            shapes=(1,i)
            shapes2=(0,i)
        used=finallist[i]/fs # sets which array from finallist will be used.
        print(len(used))
        meandurall=np.mean(used[:,1]-used[:,0])
        spikes1=[]
        res=-1
        spikes=[]
        basespk=[]
        n0,n1=0,3
        for j in range(len(used)):
            step1=[]
            step2=[]
            step3=[]
            beg= used[j][0] #Will compute the beginning of the window
            end= used[j][1] #Will compute the end of the window
            step1=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
            stepsholneg=step1[step1<0]
            step2=step1[np.where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
            step3=step1[np.where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
            spikes1+=[stepsholneg,step2,step3]
            #print(step2)
            res=res+1
            spikes2=spikes1
            spikes3=np.concatenate(spikes2[n0:n1]) # Gets the step2 and step3 arrays for scatter
            ax[shapes].scatter(spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
            n0+=3
            n1+=3
            bins=np.arange(-shoulder,meandurall+shoulder, step=binwidth)
            ax[shapes].set_xlim(min(bins), max(bins))
            ax[shapes].set_xticks([min(bins),0,meandurall,max(bins)])
            ax[shapes].tick_params(axis='both', which='major', labelsize=14)
		    #ax[shapes2].tick_params(axis='both', which='major', labelsize=18)

            normfactor=len(used)*binwidth
            ax[shapes2].set_xlim(min(bins), max(bins))
            ax[shapes2].set_title("Syllable " + sybs[i],fontsize= 18)
			#Original
            #ax[shapes2].tick_params(
            #        axis="x",          # changes apply to the x-axis
            #        which="both",      # both major and minor ticks are affected
            #        bottom=False,      # ticks along the bottom edge are off
            #        top=False,         # ticks along the top edge are off
            #        labelbottom=False)
            ax[shapes2].tick_params(
                    axis="both",          # changes apply to the x-axis
                    which="major",      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False, 
                    labelsize=14)
        # Computation of baseline
        #print("nb rendit syl")
        #print(meandurall)
        #print(normfactor)
        # Computation of baseline
        baseline_counts=[] 
        #print("\n")
        for b in range(n_baseline):
            baseline_counts_aux=0
            for j in range(len(used)):
                basecuts=np.random.choice(np.arange(basebeg,basend))
                baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
            baseline_counts+=[baseline_counts_aux/normfactor] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
		
        basemean=np.mean(baseline_counts) 
        stdbase=np.ceil(np.std(baseline_counts))
        hist_width=(int)(stdbase*10)
        baseline_counts=baseline_counts-basemean
        bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
        u,_=py.histogram(baseline_counts, bins_base,density=True)
        #py.figure()
        #py.plot(u)
		
        cumul_sig=0
        mid_hist=(int)(hist_width/hist_bin)
		#determine the level of significance for the fr (sig_fr)
		#start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
        for j in range(hist_width):
            cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
            if(cumul_sig >= 0.95):
               break
			
        sig_fr=j*hist_bin
        #print("sig_fr: ", sig_fr)
        #print("basemean: ", basemean)
        #print(meandurall)

        # b=np.sort(np.concatenate(basespk))
        # u,_= py.histogram(b, bins=np.arange(0,meandurall+binwidth,binwidth), weights=np.ones(len(b))/normfactor)
        # basemean=np.mean(u)
        # stdbase=np.std(u)
        #axis=np.arange(meandurall/3,meandurall*2/3,binwidth)
        axis=np.arange(-shoulder,meandurall+shoulder,binwidth)
        ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
        # Computation of spikes
        spikes=np.sort(np.concatenate(spikes2))
        y1,x1= py.histogram(spikes, bins=bins, weights=np.ones(len(spikes))/normfactor)
        if np.mean(y1) < 5:
            f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
        ax[shapes].axvline(x=0, color="grey", linestyle="--")
        ax[shapes].axvline(x=meandurall, color="grey", linestyle="--")
        #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
        #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
        x2=np.delete(x1,-1)
        x2=x2+binwidth/2
        ax[shapes2].plot(x2,y1, color="red")
        py.fig.subplots_adjust(hspace=0)
        #fig.subplots_adjust(hspace=0)
        black_line = mlines.Line2D([], [], color="black", label="+95%")
        black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
        green_line  = mlines.Line2D([], [], color="green", label="Mean")
        ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
        ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if len(finallist) == 1:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0,0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1,0].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        for lim in range(len(finallist)):
            values = np.array(ax[0,lim].get_ylim())
            values2 = np.array(ax[1,lim].get_ylim())
            top = np.sort(np.append(top, values))
            top2 = np.sort(np.append(top2, values2))
        for limreal in range(len(finallist)):
            ax[0,limreal].set_ylim(0,max(top))
            ax[1,limreal].set_ylim(min(top2),max(top2))        
    wind=py.get_current_fig_manager()
    #wind=plt.get_current_fig_manager()
    wind.window.showMaximized()
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109) #without this line, the psth plot is full screen but not maximally zoomed in
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0.03, "Time(seconds)", ha='center',fontsize=18)
    #plt.tight_layout()
    f.close()	
    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    #plt.savefig("PSTH.png",dpi=300, bbox_inches = "tight")
    #plt.savefig("PSTH.png")
    #py.savefig("PSTH.png")

## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles.
# Based on psth but sets different figure size (25,12) instead of (18,15). Uses linear interpolation for the firing rate in the psth
# PSTH linearly interpolated
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_interpol(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    finallist=sortsyls(motifile,0)
    #print(finallist)
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 0.05 #50 ms
    meandurall=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,len(finallist), figsize=(25,12), sharey=False)
    #fig, ax = plt.subplots(2,len(finallist), sharey=False,figsize=(25,12))
    for i in range(len(finallist)):
        if len(finallist) == 1:
            shapes = (1,)
            shapes2 = (0,)
        else:
            shapes=(1,i)
            shapes2=(0,i)
        used=finallist[i]/fs # sets which array from finallist will be used.
        print(len(used))
        meandurall=np.mean(used[:,1]-used[:,0])
        spikes1=[]
        res=-1
        spikes=[]
        basespk=[]
        n0,n1=0,3
        for j in range(len(used)):
            step1=[]
            step2=[]
            step3=[]
            beg= used[j][0] #Will compute the beginning of the window
            end= used[j][1] #Will compute the end of the window
            step1=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
            stepsholneg=step1[step1<0]
            step2=step1[np.where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
            step3=step1[np.where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
            spikes1+=[stepsholneg,step2,step3]
            #print(step2)
            res=res+1
            spikes2=spikes1
            spikes3=np.concatenate(spikes2[n0:n1]) # Gets the step2 and step3 arrays for scatter
            ax[shapes].scatter(spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
            n0+=3
            n1+=3
            bins=np.arange(-shoulder,meandurall+shoulder, step=binwidth)
            ax[shapes].set_xlim(min(bins), max(bins))
            ax[shapes].set_xticks([min(bins),0,meandurall,max(bins)])
            ax[shapes].tick_params(axis='both', which='major', labelsize=14)
		    #ax[shapes2].tick_params(axis='both', which='major', labelsize=18)

            normfactor=len(used)*binwidth
            ax[shapes2].set_xlim(min(bins), max(bins))
            ax[shapes2].set_title("Syllable " + sybs[i],fontsize= 18)
			#Original
            #ax[shapes2].tick_params(
            #        axis="x",          # changes apply to the x-axis
            #        which="both",      # both major and minor ticks are affected
            #        bottom=False,      # ticks along the bottom edge are off
            #        top=False,         # ticks along the top edge are off
            #        labelbottom=False)
            ax[shapes2].tick_params(
                    axis="both",          # changes apply to the x-axis
                    which="major",      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False, 
                    labelsize=14)
        # Computation of baseline
        #print("nb rendit syl")
        #print(meandurall)
        #print(normfactor)
        # Computation of baseline
        baseline_counts=[] 
        #print("\n")
        for b in range(n_baseline):
            baseline_counts_aux=0
            for j in range(len(used)):
                basecuts=np.random.choice(np.arange(basebeg,basend))
                baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
            baseline_counts+=[baseline_counts_aux/normfactor] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
		
        basemean=np.mean(baseline_counts) 
        stdbase=np.ceil(np.std(baseline_counts))
        hist_width=(int)(stdbase*10)
        baseline_counts=baseline_counts-basemean
        bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
        u,_=py.histogram(baseline_counts, bins_base,density=True)
        #py.figure()
        #py.plot(u)
		
        cumul_sig=0
        mid_hist=(int)(hist_width/hist_bin)
		#determine the level of significance for the fr (sig_fr)
		#start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
        for j in range(hist_width):
            cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
            if(cumul_sig >= 0.95):
               break
			
        sig_fr=j*hist_bin
        #print("sig_fr: ", sig_fr)
        #print("basemean: ", basemean)
        #print(meandurall)

        # b=np.sort(np.concatenate(basespk))
        # u,_= py.histogram(b, bins=np.arange(0,meandurall+binwidth,binwidth), weights=np.ones(len(b))/normfactor)
        # basemean=np.mean(u)
        # stdbase=np.std(u)
        #axis=np.arange(meandurall/3,meandurall*2/3,binwidth)
        axis=np.arange(meandurall/5,meandurall*4/5,binwidth)
        ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
        # Computation of spikes
        spikes=np.sort(np.concatenate(spikes2))
        y1,x1= py.histogram(spikes, bins=bins, weights=np.ones(len(spikes))/normfactor)
        if np.mean(y1) < 5:
            f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
        ax[shapes].axvline(x=0, color="grey", linestyle="--")
        ax[shapes].axvline(x=meandurall, color="grey", linestyle="--")
        #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
        #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
        x2=np.delete(x1,-2)
        x2[1:-1]=x2[1:-1]+binwidth/2
        inter = scipy.interpolate.interp1d(x2, y1, kind="linear")
        xnew=np.linspace(min(x2),max(x2), num=100)
        ax[shapes2].plot(xnew,inter(xnew), color="red")
        py.fig.subplots_adjust(hspace=0)
        #fig.subplots_adjust(hspace=0)
        black_line = mlines.Line2D([], [], color="black", label="+95%")
        black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
        green_line  = mlines.Line2D([], [], color="green", label="Mean")
        ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
        ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if len(finallist) == 1:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0,0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1,0].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        for lim in range(len(finallist)):
            values = np.array(ax[0,lim].get_ylim())
            values2 = np.array(ax[1,lim].get_ylim())
            top = np.sort(np.append(top, values))
            top2 = np.sort(np.append(top2, values2))
        for limreal in range(len(finallist)):
            ax[0,limreal].set_ylim(0,max(top))
            ax[1,limreal].set_ylim(min(top2),max(top2))        
    wind=py.get_current_fig_manager()
    #wind=plt.get_current_fig_manager()
    wind.window.showMaximized()
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109) #without this line, the psth plot is full screen but not maximally zoomed in
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0.03, "Time(seconds)", ha='center',fontsize=18)
    #plt.tight_layout()
    f.close()	
    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    #plt.savefig("PSTH.png",dpi=300, bbox_inches = "tight")
    #plt.savefig("PSTH.png")
    #py.savefig("PSTH.png")	
## 
#
# This function generates a PSTH for motifs of playback songs. 
# To be used with the new matfiles.
# Based on psth but uses the interval of 2 seconds before the motif onset as the baseline 
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency 
# 
# n is the index of the BOS (n=1), rBOS (n=2) or OCS (n=3)
def psth_pb(spikefile, motifile,n,binwidth=binwidth, fs=fs):      
    #sybs_pb=["A","B","C"]
    #sybs_pb=["rC","rB","rA"]
    #sybs_pb=["S","T","U","V","W"]
    if(n==1): #BOS
       sybs_=sybs
    elif(n==2): #rBOS
       sybs_=sybs_rBOS
    else: #OCS
       sybs_=sybs_OCS

    finallist=sortsyls(motifile,n)
    baseline_pb=baseline_playback(motifile)
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 0.05 #50 ms
    meandurall=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,len(finallist), figsize=(25,12), sharey=False)
    #fig, ax = plt.subplots(2,len(finallist), sharey=False,figsize=(25,12))
    for i in range(len(finallist)):
        if len(finallist) == 1:
            shapes = (1,)
            shapes2 = (0,)
        else:
            shapes=(1,i)
            shapes2=(0,i)
        used=finallist[i]/fs # sets which array from finallist will be used.
        print(len(used))
        meandurall=np.mean(used[:,1]-used[:,0])
        spikes1=[]
        res=-1
        spikes=[]
        basespk=[]
        n0,n1=0,3
        for j in range(len(used)):
            step1=[]
            step2=[]
            step3=[]
            beg= used[j][0] #Will compute the beginning of the window
            end= used[j][1] #Will compute the end of the window
            step1=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
            stepsholneg=step1[step1<0]
            step2=step1[np.where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
            step3=step1[np.where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
            spikes1+=[stepsholneg,step2,step3]
            #print(step2)
            res=res+1
            spikes2=spikes1
            spikes3=np.concatenate(spikes2[n0:n1]) # Gets the step2 and step3 arrays for scatter
            ax[shapes].scatter(spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
            n0+=3
            n1+=3
            bins=np.arange(-shoulder,meandurall+shoulder, step=binwidth)
            ax[shapes].set_xlim(min(bins), max(bins))
            ax[shapes].set_xticks([min(bins),0,meandurall,max(bins)])
            ax[shapes].tick_params(axis='both', which='major', labelsize=14)
		    #ax[shapes2].tick_params(axis='both', which='major', labelsize=18)

            normfactor=len(used)*binwidth
            ax[shapes2].set_xlim(min(bins), max(bins))
            if(n==2): #rBOS
               ax[shapes2].set_title("Syllable " + "r" + sybs[len(finallist)-1-i],fontsize= 18)
            else:
               ax[shapes2].set_title("Syllable " + sybs_[i],fontsize= 18)
			#Original
            #ax[shapes2].tick_params(
            #        axis="x",          # changes apply to the x-axis
            #        which="both",      # both major and minor ticks are affected
            #        bottom=False,      # ticks along the bottom edge are off
            #        top=False,         # ticks along the top edge are off
            #        labelbottom=False)
            ax[shapes2].tick_params(
                    axis="both",          # changes apply to the x-axis
                    which="major",      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False, 
                    labelsize=14)
        # Computation of baseline
        #print("nb rendit syl")
        #print(meandurall)
        #print(normfactor)
		###############################################
        # Computation of baseline_counts
		###############################################
        baseline_counts=[] 
        len_baseline_pb=len(baseline_pb[0]) #nb of baseline chuncks
        baseline_width=0.2
        #print(len_baseline_pb)
        for b in range(n_baseline):
            baseline_counts_aux=0
            #print(b)
            for j in range(len(used)):
                idx_baseline=np.random.choice(np.arange(0,len_baseline_pb)) #Take randomly a baseline chunk
                basebeg_pb=baseline_pb[0][idx_baseline]/fs
                basend_pb=baseline_pb[1][idx_baseline]/fs
                basecuts=np.random.choice(np.arange(basebeg_pb,basend_pb,baseline_width))
                baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
            baseline_counts+=[baseline_counts_aux/normfactor] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
		
        #print(baseline_counts)
        basemean=np.mean(baseline_counts) 
        stdbase=np.ceil(np.std(baseline_counts))
        hist_width=(int)(stdbase*10)
        baseline_counts=baseline_counts-basemean
        bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
        u,_=py.histogram(baseline_counts, bins_base,density=True)
        #py.figure()
        #py.plot(u)
		
        cumul_sig=0
        mid_hist=(int)(hist_width/hist_bin)
		#determine the level of significance for the fr (sig_fr)
		#start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
        for j in range(hist_width):
            cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
            if(cumul_sig >= 0.95):
               break
			
        sig_fr=j*hist_bin
        #print("sig_fr: ", sig_fr)
        #print("basemean: ", basemean)
        #print(meandurall)

        # b=np.sort(np.concatenate(basespk))
        # u,_= py.histogram(b, bins=np.arange(0,meandurall+binwidth,binwidth), weights=np.ones(len(b))/normfactor)
        # basemean=np.mean(u)
        # stdbase=np.std(u)
        #axis=np.arange(meandurall/3,meandurall*2/3,binwidth)
        axis=np.arange(-shoulder,meandurall+shoulder,binwidth)
        ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
        # Computation of spikes
        spikes=np.sort(np.concatenate(spikes2))
        y1,x1= py.histogram(spikes, bins=bins, weights=np.ones(len(spikes))/normfactor)
        if np.mean(y1) < 5:
            f.writelines("Syllable " + str(sybs_[i]) +" : " + str(np.mean(y1)) + "\n")
        ax[shapes].axvline(x=0, color="grey", linestyle="--")
        ax[shapes].axvline(x=meandurall, color="grey", linestyle="--")
        #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
        #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
        x2=np.delete(x1,-1)
        x2=x2+binwidth/2
        ax[shapes2].plot(x2,y1, color="red")
        py.fig.subplots_adjust(hspace=0)
        #fig.subplots_adjust(hspace=0)
        black_line = mlines.Line2D([], [], color="black", label="+95%")
        black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
        green_line  = mlines.Line2D([], [], color="green", label="Mean")
        ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
        ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if len(finallist) == 1:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0,0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1,0].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        for lim in range(len(finallist)):
            values = np.array(ax[0,lim].get_ylim())
            values2 = np.array(ax[1,lim].get_ylim())
            top = np.sort(np.append(top, values))
            top2 = np.sort(np.append(top2, values2))
        for limreal in range(len(finallist)):
            ax[0,limreal].set_ylim(0,max(top))
            ax[1,limreal].set_ylim(min(top2),max(top2))        
    wind=py.get_current_fig_manager()
    #wind=plt.get_current_fig_manager()
    wind.window.showMaximized()
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109) #without this line, the psth plot is full screen but not maximally zoomed in
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0, "Time(seconds)", va="center", ha="center",fontsize=18)
    #fig.text(0.5, 0.03, "Time(seconds)", ha='center',fontsize=18)
    #plt.tight_layout()
    f.close()	
    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    #plt.savefig("PSTH.png",dpi=300, bbox_inches = "tight")
    #plt.savefig("PSTH.png")
    #py.savefig("PSTH.png")	
	
	
## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles.
# Based on psth but for White noise playback (no DTW, ...). One PSTH for noise onset and one for noise offset
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency 
def psth_wn(spikefile, motifile,binwidth=binwidth, fs=fs):      
    finallist=sortsyls_wn(motifile)
    baseline_pb=baseline_playback(motifile)
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 1 #1s
    meandurall=1 #800 ms
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,2, figsize=(25,12), sharey=False)
	
    shapes_on=(1,0)
    shapes2_on=(0,0)	
    shapes_off=(1,1)
    shapes2_off=(0,1)	
	
    #meandurall=np.mean(used[:,1]-used[:,0])
    spikes_on=[]
    spikes_off=[]
    res=-1
    n0=0
    used=np.array(finallist)/fs # sets which array from finallist will be used.

	
    for i in range(len(used)):
	    #Onset and Offset
        beg= used[i][0] #Will compute the beginning of the window
        end= used[i][1] #Will compute the end of the window
        onset=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= beg+meandurall) == True)]-beg #origin at onset of noise
        offset=spused[np.where(np.logical_and(spused >= end-meandurall, spused <= end+shoulder) == True)]-end #origin at offset of noise
        spikes_on+=[onset]
        spikes_off+=[offset]
        #print(step2)
        res=res+1
        ax[shapes_on].scatter(spikes_on[n0],res+np.zeros(len(spikes_on[n0])),marker="|", color="black")
        ax[shapes_off].scatter(spikes_off[n0],res+np.zeros(len(spikes_off[n0])),marker="|", color="black")
        n0+=1
        
    normfactor=len(used)*binwidth 
    ##################################################    
    #          Computation of baseline
	##################################################
    baseline_counts=[] 
    len_baseline_pb=len(baseline_pb[0]) #nb of baseline chuncks
    baseline_width=0.4
    #print(len_baseline_pb)
    for b in range(n_baseline):
        baseline_counts_aux=0
        #print(b)
        for j in range(len(used)):
            idx_baseline=np.random.choice(np.arange(0,len_baseline_pb)) #Take randomly a baseline chunk
            basebeg_pb=baseline_pb[0][idx_baseline]/fs
            basend_pb=baseline_pb[1][idx_baseline]/fs
            basecuts=np.random.choice(np.arange(basebeg_pb,basend_pb,baseline_width))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
		
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)

    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
	
    sig_fr=j*hist_bin
    #print(sig_fr)

    # b=np.sort(np.concatenate(basespk))
    # u,_= py.histogram(b, bins=np.arange(0,meandurall+binwidth,binwidth), weights=np.ones(len(b))/normfactor)
    # basemean=np.mean(u)
    # stdbase=np.std(u)
	
    bins_on=np.arange(-shoulder,meandurall, step=binwidth)
    bins_off=np.arange(-meandurall,shoulder, step=binwidth)
    
    ax[shapes_on].set_xlim(min(bins_on), max(bins_on))
    ax[shapes_on].set_xticks([min(bins_on),0,max(bins_on)])
    ax[shapes_on].tick_params(axis='both', which='major', labelsize=14)
    #ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes_off].set_xlim(min(bins_off), max(bins_off))
    ax[shapes_off].set_xticks([min(bins_off),0,max(bins_off)])
    ax[shapes_off].tick_params(axis='both', which='major', labelsize=14)
    
    
    ax[shapes2_on].set_xlim(min(bins_on), max(bins_on))
    ax[shapes2_on].set_title("W noise onset ",fontsize= 18)
    
    ax[shapes2_off].set_xlim(min(bins_off), max(bins_off))
    ax[shapes2_off].set_title("W noise offset ",fontsize= 18)
    #Original
    #ax[shapes2].tick_params(
    #        axis="x",          # changes apply to the x-axis
    #        which="both",      # both major and minor ticks are affected
    #        bottom=False,      # ticks along the bottom edge are off
    #        top=False,         # ticks along the top edge are off
    #        labelbottom=False)
    ax[shapes2_on].tick_params(
            axis="both",          # changes apply to the x-axis
            which="major",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False, 
            labelsize=14)
    
    ax[shapes2_off].tick_params(
            axis="both",          # changes apply to the x-axis
            which="major",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False, 
            labelsize=14)	
    
	
	
    axis=np.arange(-shoulder,meandurall,binwidth)
    ax[shapes2_on].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2_on].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2_on].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")

    axis=np.arange(-meandurall,shoulder,binwidth)
    ax[shapes2_off].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2_off].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2_off].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")

    # Computation of spikes_on
    spikes_on=np.sort(np.concatenate(spikes_on))
    y1,x1= py.histogram(spikes_on, bins=bins_on, weights=np.ones(len(spikes_on))/normfactor)
    if np.mean(y1) < 5:
        f.writelines("Syllable z : " + str(np.mean(y1)) + "\n")

    #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
    #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2_on].plot(x2,y1, color="red")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2_on].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes_on].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # Computation of spikes_off
    spikes_off=np.sort(np.concatenate(spikes_off))
    y1,x1= py.histogram(spikes_off, bins=bins_off, weights=np.ones(len(spikes_off))/normfactor)
    if np.mean(y1) < 5:
        f.writelines("Syllable z : " + str(np.mean(y1)) + "\n")

    #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
    #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2_off].plot(x2,y1, color="red")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2_off].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes_off].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))		


    ax[shapes_on].axvline(x=0, color="grey", linestyle="--")
    ax[shapes_off].axvline(x=0, color="grey", linestyle="--")
    ax[shapes2_on].axvline(x=0, color="grey", linestyle="--")
    ax[shapes2_off].axvline(x=0, color="grey", linestyle="--")

    ax[shapes2_on].set_ylabel("Spikes/Sec",fontsize=18)
    ax[shapes_on].set_ylabel("Motif number",fontsize=18)
	
    ax[shapes2_off].set_ylabel("Spikes/Sec",fontsize=18)
    ax[shapes_off].set_ylabel("Motif number",fontsize=18)
     

    values = np.array([])
    values2 = np.array([])
    top = np.array([])
    top2 = np.array([])
    for lim in range(len(finallist[0])):
        values = np.array(ax[0,lim].get_ylim())
        values2 = np.array(ax[1,lim].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
    for limreal in range(len(finallist[0])):
        ax[0,limreal].set_ylim(0,max(top))
        ax[1,limreal].set_ylim(min(top2),max(top2))        
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()	 


## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth() but gives a single psth for the whole motif under probabilistic noise feedback
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot. Should not be called (gives error) if no noisy renditions of the syllable, in that
# case call instead psth_glob_sep_no_noise. PSTH linearly interpolated
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_interpol(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    all_motifs=np.concatenate((np.array(finallist[1]),np.array(finallist[0])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           #print(used_off)
           #print("\n")
           #print(used_on)
           #print("\n")
           #print("\n")
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           #print(meandurall)
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           #print(used_off)
           #print("\n")
           #print(used_on)
           #print("\n")
           #print("\n")
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth
	
    #print(meandurall_list)

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)

    py.fig.text(0.17, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    py.fig.text(0.38, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    py.fig.text(0.73, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 


    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           used_off=all_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=all_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[0])
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[0])	   

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           used_on=all_motifs[:,i] # sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=all_motifs[:,i]
           used_off=(used_off/fs)+shoulder_end # considers the time delay due to shoulder end
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           shift_syl_plot=shift_syl_plot+shoulder_end
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           if(idx_noisy_syb==len_motif-1):#last syl is the one that is targeted wih noise
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))	
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))
           else:
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])   

        elif(i!=2*idx_noisy_syb):
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
        
		#treat the spikes within the syllable, for the syllable that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-1])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-2)
    x2[1:-1]=x2[1:-1]+binwidth/2
    xnew=np.linspace(min(x2),max(x2), num=400)
	
    inter_cln = scipy.interpolate.interp1d(x2, y_cln, kind="linear")
    inter_noisy = scipy.interpolate.interp1d(x2, y_noisy, kind="linear")
    #xnew=np.linspace(min(x2),max(x2), num=100)
    #inter = scipy.interpolate.interp1d(x2, y1, kind="linear")
    ynew_cln=inter_cln(xnew)
    ynew_noisy=inter_noisy(xnew)

    ax[shapes2].plot(xnew,ynew_noisy, color="blue")
    ax[shapes2].plot(xnew,ynew_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   		

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()

	
## 
#
# Based on psth_glob_interpol but without interpolation
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in sybs of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    all_motifs=np.concatenate((np.array(finallist[1]),np.array(finallist[0])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth
	
    #print(meandurall_list)

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)

    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           used_off=all_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=all_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[0])
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[0])	   

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           used_on=all_motifs[:,i] # sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=all_motifs[:,i]
           used_off=(used_off/fs)+shoulder_end # considers the time delay due to shoulder end
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           shift_syl_plot=shift_syl_plot+shoulder_end
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           if(idx_noisy_syb==len_motif-1):#last syl is the one that is targeted wih noise
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))	
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))
           else:
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])   

        elif(i!=2*idx_noisy_syb):
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
        
		#treat the spikes within the syllable, for the syllable that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-1])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")	
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   		

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()

	
## 
#
# Based on psth_glob but for the case the white noise lasts more than the syllable. Should never happen after 30.09.2020 because the psth is then not correct
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_long_noise(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
	
    all_motifs=np.concatenate((np.array(finallist[1]),np.array(finallist[0])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           #meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth

    #correct the noisy syllable: change length to average length of clean renditions
    noisy_motifs[:,2*idx_noisy_syb+1]=noisy_motifs[:,2*idx_noisy_syb]+meandurall_list[2*idx_noisy_syb]*fs
    all_motifs=np.concatenate((noisy_motifs,clean_motifs),axis=0)
	
    #print(meandurall_list)

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
	
	
	##############################################################
    ##x_ticks.append(min(bins))
    #x_ax_len=shoulder_beg
    #for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
    #    x_ticks.append(x_ax_len)
    #    x_ax_len=x_ax_len+meandurall_list[i]
    #x_ticks.append(x_ax_len)
    #x_ticks.append(x_ax_len+shoulder_end)
	###############################################################
	######################################################
	#X ticks: only onset of syllables
	######################################################
    x_ax_len=shoulder_beg
    for i in range(len_motif): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        if(i!=(len_motif-1)):
            x_ax_len=x_ax_len+meandurall_list[2*i]+meandurall_list[2*i+1]
        else:
            x_ax_len=x_ax_len+meandurall_list[2*i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
	######################################################	
		
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)

    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           used_off=all_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=all_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[0])
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[0])	   

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           used_on=all_motifs[:,i] # sets the onsets of first sylable in motif
           used_on=used_on/fs
           used_off=all_motifs[:,i]
           used_off=(used_off/fs)+shoulder_end # considers the time delay due to shoulder end
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           shift_syl_plot=shift_syl_plot+shoulder_end
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           if(idx_noisy_syb==len_motif-1):#last syl is the one that is targeted wih noise
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))	
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))
           else:
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])   

        elif(i!=2*idx_noisy_syb):
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
        
		#treat the spikes within the syllable, for the syllable that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-1])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")	
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   		

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()


	
## 
#
# Based on psth_glob_long_noise but noisy and clean syllables are put together in the psth
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_long_noise_mixed(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
	
    all_motifs=np.concatenate((np.array(finallist[1]),np.array(finallist[0])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
            used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=all_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_on=used_on/fs
            used_off=used_off/fs
            
            meandurall=np.mean(used_off[:]-used_on[:])
            normfactor[i]=len(used_off[:])
            meandurall_list[i]=meandurall
            normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		    #clean versions of the syll
            used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_on=used_on/fs
            used_off=used_off/fs
            
            meandurall=np.mean(used_off[:]-used_on[:])
            #n_clean=len(used_off[:])
            meandurall_list[i]=meandurall
            normfactor[i]=len(used_off[:])
		    
		    #noisy versions of the syll
            used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            
            used_on=used_on/fs
            used_off=used_off/fs
            
            #meandurall=np.mean(used_off[:]-used_on[:])
            #n_noisy=len(used_off[:])
            #meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
            #normfactor[-1]=len(used_off[:])
            normfactor[i]=normfactor[i]+len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth

    #correct the noisy syllable: change length to average length of clean renditions
    noisy_motifs[:,2*idx_noisy_syb+1]=noisy_motifs[:,2*idx_noisy_syb]+meandurall_list[2*idx_noisy_syb]*fs
    all_motifs=np.concatenate((noisy_motifs,clean_motifs),axis=0)
	
    #print(meandurall_list)

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)

    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	################################################
	#    Do DTW: treat all syllables and gaps
	################################################
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           used_off=all_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=all_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[0])
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[0])	   

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           used_on=all_motifs[:,i] # sets the onsets of first sylable in motif
           used_on=used_on/fs
           used_off=all_motifs[:,i]
           used_off=(used_off/fs)+shoulder_end # considers the time delay due to shoulder end
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           shift_syl_plot=shift_syl_plot+shoulder_end
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           if(idx_noisy_syb==len_motif-1):#last syl is the one that is targeted wih noise
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))	
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))
           else:
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])   

        else: #elif(i!=2*idx_noisy_syb):
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
        
		# Not needed since no discrimination between noisy and clean syls
		## #treat the spikes within the syllable, for the syllable that is targeted with noise	   
        ## else:	
        ##    #noisy renditions	    
        ##    used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
        ##    #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
        ##    used_on=used_on/fs
        ##    used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
        ##    #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        ##    used_off=used_off/fs
        ##    spikes1=[]
        ##    res=-1
        ##    n0=0
        ##    for j in range(len(used_on)):
        ##        step1=[]
        ##        step2=[]
        ##        step3=[]
        ##        beg= used_on[j] #Will compute the beginning of the window
        ##        end= used_off[j] #Will compute the end of the window
        ##        if(beg>0 and end>0):#syllable sung
        ##           step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
        ##           step2=step1*(meandurall_list[i]/(end-beg))
        ##           spikes1+=[step2]
        ##           res=res+1
        ##           #spikes2_noisy=spikes1
        ##           spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		##    	      #Shift the spike times for each syllable type for the scatter plot 
        ##           ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
        ##           spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
        ##           n0+=1
        ##        else:#syllable not sung
        ##           res=res+1 
        ##    #bins_edge=bins_edge+meandurall_list[i]		
        ##    #shift_syl_plot=shift_syl_plot+meandurall_list[i]
        ##    #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
        ##    #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
        ##    weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-1])
		## 		  
        ##    #clean renditions	  
        ##    used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        ##    #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
        ##    used_on=used_on/fs
        ##    used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        ##    #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        ##    used_off=used_off/fs
        ##    spikes1=[]
        ##    #res=-1 continue with previous value of res
        ##    spikes_cln=[]
        ##    n0=0
        ##    for j in range(len(used_on)):
        ##        step1=[]
        ##        step2=[]
        ##        step3=[]
        ##        beg= used_on[j] #Will compute the beginning of the window
        ##        end= used_off[j] #Will compute the end of the window
        ##        if(beg>0 and end>0):#syllable sung
        ##           step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
        ##           step2=step1*(meandurall_list[i]/(end-beg))
        ##           spikes1+=[step2]
        ##           res=res+1
        ##           #spikes2_cln=spikes1
        ##           spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		##    	      #Shift the spike times for each syllable type for the scatter plot 
        ##           ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
        ##           spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
        ##           n0+=1 
        ##        else:#syllable not sung
        ##           res=res+1
        ##    bins_edge=bins_edge+meandurall_list[i]		
        ##    shift_syl_plot=shift_syl_plot+meandurall_list[i]
        ##    ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
        ##    ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
        ##    weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    #spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    #y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    #ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")	
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   		

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()
    
	
    return y_cln
	
	
## 
#
# Based on psth_glob_long_noise_mixed but in case the noise is not longer than the syllable
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_mixed(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
	
    all_motifs=np.concatenate((np.array(finallist[1]),np.array(finallist[0])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
            used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=all_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_on=used_on/fs
            used_off=used_off/fs
            
            meandurall=np.mean(used_off[:]-used_on[:])
            normfactor[i]=len(used_off[:])
            meandurall_list[i]=meandurall
            normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		    #clean versions of the syll
            used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_on=used_on/fs
            used_off=used_off/fs
            
            meandurall_cln=np.mean(used_off[:]-used_on[:])
            n_clean=len(used_off[:])
            #meandurall_list[i]=meandurall
            normfactor[i]=len(used_off[:])
		    
		    #noisy versions of the syll
            used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
            used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
            used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
            
            used_on=used_on/fs
            used_off=used_off/fs
            
            meandurall_noisy=np.mean(used_off[:]-used_on[:])
            n_noisy=len(used_off[:])
            meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_cln+(n_noisy/(n_clean+n_noisy))*meandurall_noisy) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
            #normfactor[-1]=len(used_off[:])
            normfactor[i]=normfactor[i]+len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth

    #correct the noisy syllable: change length to average length of clean renditions
    #noisy_motifs[:,2*idx_noisy_syb+1]=noisy_motifs[:,2*idx_noisy_syb]+meandurall_list[2*idx_noisy_syb]*fs
    #all_motifs=np.concatenate((noisy_motifs,clean_motifs),axis=0)
	
    #print(meandurall_list)

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)

    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	################################################
	#    Do DTW: treat all syllables and gaps
	################################################
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           used_off=all_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=all_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[0])
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[0])	   

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           used_on=all_motifs[:,i] # sets the onsets of first sylable in motif
           used_on=used_on/fs
           used_off=all_motifs[:,i]
           used_off=(used_off/fs)+shoulder_end # considers the time delay due to shoulder end
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           shift_syl_plot=shift_syl_plot+shoulder_end
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           if(idx_noisy_syb==len_motif-1):#last syl is the one that is targeted wih noise
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))	
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/(normfactor[-1]+normfactor[-2]))
           else:
              weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])
              weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-2])   

        else: #elif(i!=2*idx_noisy_syb):
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	
        
		# Not needed since no discrimination between noisy and clean syls
		## #treat the spikes within the syllable, for the syllable that is targeted with noise	   
        ## else:	
        ##    #noisy renditions	    
        ##    used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
        ##    #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
        ##    used_on=used_on/fs
        ##    used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
        ##    #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        ##    used_off=used_off/fs
        ##    spikes1=[]
        ##    res=-1
        ##    n0=0
        ##    for j in range(len(used_on)):
        ##        step1=[]
        ##        step2=[]
        ##        step3=[]
        ##        beg= used_on[j] #Will compute the beginning of the window
        ##        end= used_off[j] #Will compute the end of the window
        ##        if(beg>0 and end>0):#syllable sung
        ##           step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
        ##           step2=step1*(meandurall_list[i]/(end-beg))
        ##           spikes1+=[step2]
        ##           res=res+1
        ##           #spikes2_noisy=spikes1
        ##           spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		##    	      #Shift the spike times for each syllable type for the scatter plot 
        ##           ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
        ##           spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
        ##           n0+=1
        ##        else:#syllable not sung
        ##           res=res+1 
        ##    #bins_edge=bins_edge+meandurall_list[i]		
        ##    #shift_syl_plot=shift_syl_plot+meandurall_list[i]
        ##    #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
        ##    #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
        ##    weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor[-1])
		## 		  
        ##    #clean renditions	  
        ##    used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        ##    #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
        ##    used_on=used_on/fs
        ##    used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        ##    #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        ##    used_off=used_off/fs
        ##    spikes1=[]
        ##    #res=-1 continue with previous value of res
        ##    spikes_cln=[]
        ##    n0=0
        ##    for j in range(len(used_on)):
        ##        step1=[]
        ##        step2=[]
        ##        step3=[]
        ##        beg= used_on[j] #Will compute the beginning of the window
        ##        end= used_off[j] #Will compute the end of the window
        ##        if(beg>0 and end>0):#syllable sung
        ##           step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
        ##           step2=step1*(meandurall_list[i]/(end-beg))
        ##           spikes1+=[step2]
        ##           res=res+1
        ##           #spikes2_cln=spikes1
        ##           spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		##    	      #Shift the spike times for each syllable type for the scatter plot 
        ##           ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
        ##           spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
        ##           n0+=1 
        ##        else:#syllable not sung
        ##           res=res+1
        ##    bins_edge=bins_edge+meandurall_list[i]		
        ##    shift_syl_plot=shift_syl_plot+meandurall_list[i]
        ##    ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
        ##    ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
        ##    weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    #spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    #y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    #ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")	
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   		

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()
    
	
    return y_cln
	
	
	
## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth_glob() but gives two superimposed psths: one for motifs under probabilistic noise feedback and the other for clean motifs
# If no probabilistic noise feedback, use this function by setting idx_noisy_syb to some irrelevant value (for example -2) that is not caught by the if conditions in the code
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot. PSTH linearly interpolated
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_sep_interpol(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
	
    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for clean and noisy motifs
    meandurall_list_clean = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_clean = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    meandurall_list_noisy = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_noisy = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):       
		#clean versions of the syll
        used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_clean=len(used_off[:])
        meandurall_list_clean[i]=meandurall
        normfactor_clean[i]=len(used_off[:])
		
		#noisy versions of the syll
        used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_noisy=len(used_off[:])
        meandurall_list_noisy[i]=meandurall #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
        normfactor_noisy[i]=len(used_off[:])
		   
    normfactor_mean_clean=np.mean(normfactor_clean)
    mean_nb_rendit_syl_clean=normfactor_mean_clean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_clean=normfactor_mean_clean*binwidth
    normfactor_clean=normfactor_clean*binwidth
	
    normfactor_mean_noisy=np.mean(normfactor_noisy)
    mean_nb_rendit_syl_noisy=normfactor_mean_noisy #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_noisy=normfactor_mean_noisy*binwidth
    normfactor_noisy=normfactor_noisy*binwidth

    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for all motifs
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth	
	

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)
	
    #py.fig.text(0.17, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.38, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.73, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 

    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      
	
    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	
	
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_off=noisy_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=noisy_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1  			   
   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[0])	 
		   

		   #clean
           spikes1=[]
           n0=0
           used_off=clean_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=clean_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               n0+=1   

           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[0]) 

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[-1])	 
		   
		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           #shift_syl_plot=shift_syl_plot+shoulder_end
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[-1]) 


        elif(i!=2*idx_noisy_syb):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])	

		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
	
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i]) 

		
		#treat the spikes within the syllable, that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-2)
    x2[1:-1]=x2[1:-1]+binwidth/2
    xnew=np.linspace(min(x2),max(x2), num=400)
	
    inter_cln = scipy.interpolate.interp1d(x2, y_cln, kind="linear")
    inter_noisy = scipy.interpolate.interp1d(x2, y_noisy, kind="linear")
    #xnew=np.linspace(min(x2),max(x2), num=100)
    #inter = scipy.interpolate.interp1d(x2, y1, kind="linear")
    ynew_cln=inter_cln(xnew)
    ynew_noisy=inter_noisy(xnew)

    ax[shapes2].plot(xnew,ynew_noisy, color="blue")
    ax[shapes2].plot(xnew,ynew_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
	

    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2)) 
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()

## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth_glob() but gives two superimposed psths: one for motifs under probabilistic noise feedback and the other for clean motifs
# If no probabilistic noise feedback, use this function by setting idx_noisy_syb to some irrelevant value (for example -2) that is not caught by the if conditions in the code
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot.
# Based on psth_glob_sep_interpol but without linear interpolation
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_sep(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    #print(clean_motifs)
    #print(noisy_motifs)
    all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
	
    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for clean and noisy motifs
    meandurall_list_clean = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_clean = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    meandurall_list_noisy = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_noisy = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):       
		#clean versions of the syll
        used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        #print(used_off)
        #print("\n")
        #print(used_on)
        #print("\n")
        #print("\n")
        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_clean=len(used_off[:])
        meandurall_list_clean[i]=meandurall
        normfactor_clean[i]=len(used_off[:])
		
		#noisy versions of the syll
        used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_noisy=len(used_off[:])
        meandurall_list_noisy[i]=meandurall #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
        normfactor_noisy[i]=len(used_off[:])
		   
    
    normfactor_mean_clean=np.mean(normfactor_clean)
    mean_nb_rendit_syl_clean=normfactor_mean_clean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_clean=normfactor_mean_clean*binwidth
    normfactor_clean=normfactor_clean*binwidth
	
    normfactor_mean_noisy=np.mean(normfactor_noisy)
    mean_nb_rendit_syl_noisy=normfactor_mean_noisy #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_noisy=normfactor_mean_noisy*binwidth
    normfactor_noisy=normfactor_noisy*binwidth

    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for all motifs
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
           if(len_motif == 1): # only one syllable and it is the one receiving porbabilistic noise
             c_m=clean_motifs[:,0]
             n_m=noisy_motifs[:,0]
             normfactor_mean=((n_clean/(n_clean+n_noisy))*len(c_m[:])+(n_noisy/(n_clean+n_noisy))*len(n_m[:]))

	   
    mean_nb_rendit_syl=int(normfactor_mean) #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth	
	

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)
	
    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    #py.fig.text(0.17, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.38, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.73, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	
	
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_off=noisy_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=noisy_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1  			   
   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[0])	 
		   

		   #clean
           spikes1=[]
           n0=0
           used_off=clean_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=clean_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               n0+=1   

           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[0]) 

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[-1])	 
		   
		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           #shift_syl_plot=shift_syl_plot+shoulder_end
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[-1]) 


        elif(i!=2*idx_noisy_syb):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])	

		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
	
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i]) 

		
		#treat the spikes within the syllable, that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
	

    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2)) 
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()


## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth_glob() but gives two superimposed psths: one for motifs under probabilistic noise feedback and the other for clean motifs
# If no probabilistic noise feedback, use this function by setting idx_noisy_syb to some irrelevant value (for example -2) that is not caught by the if conditions in the code
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot.
# Based on psth_glob_sep but for the case, the white noise lasts too long and thus makes the noisy versions of the syllables longer than the clean ones
# In that case, cut the noisy syllable at the average length of the clean ones
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth_glob_sep_long_noise(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C","D"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb. if sybs=["a","b","c","d"] and the syllable receiving noise is c (and d is thus the noisy version of c), then idx_noisy_syb = 2
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)
    #nb_syls=len(sybs) #number of syllables, the noisy syllable is considered as an additional syllable
	
    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
	
    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for clean and noisy motifs
    meandurall_list_clean = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_clean = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    meandurall_list_noisy = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_noisy = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):       
		#clean versions of the syll
        used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        #print(used_off)
        #print("\n")
        #print(used_on)
        #print("\n")
        #print("\n")
        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_clean=len(used_off[:])
        meandurall_list_clean[i]=meandurall
        normfactor_clean[i]=len(used_off[:])
		
		#noisy versions of the syll
        used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_noisy=len(used_off[:])
        if(i==2*idx_noisy_syb): #syllable targeted with noise
          meandurall_list_noisy[i]=meandurall_list_clean[i] #mean of the noisy and clean renditions of the syllable. Makes more sense to do DTW in the same way for noisy and clean syllables
        else:
          meandurall_list_noisy[i]=meandurall #mean of the noisy and clean renditions of the syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
        normfactor_noisy[i]=len(used_off[:])
		   
    
    normfactor_mean_clean=np.mean(normfactor_clean)
    mean_nb_rendit_syl_clean=normfactor_mean_clean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_clean=normfactor_mean_clean*binwidth
    normfactor_clean=normfactor_clean*binwidth
	
    normfactor_mean_noisy=np.mean(normfactor_noisy)
    mean_nb_rendit_syl_noisy=normfactor_mean_noisy #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_noisy=normfactor_mean_noisy*binwidth
    normfactor_noisy=normfactor_noisy*binwidth

    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for all motifs
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
        if(i!=2*idx_noisy_syb):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
        else: #the syllable receiving contingent noise
		   #clean versions of the syll
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_clean=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor[i]=len(used_off[:])
		   
		   #noisy versions of the syll
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung

           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           n_noisy=len(used_off[:])
           #meandurall_list[i]=((n_clean/(n_clean+n_noisy))*meandurall_list[i]+(n_noisy/(n_clean+n_noisy))*meandurall) #mean of the noisy and clean renditions of he syllable. Makes more sense to do DTW in the same way for noisy and clean syllables 
           normfactor[-1]=len(used_off[:])
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth	
	
    #correct the noisy syllable: change length to average length of clean renditions
    noisy_motifs[:,2*idx_noisy_syb+1]=noisy_motifs[:,2*idx_noisy_syb]+meandurall_list[2*idx_noisy_syb]*fs
    #all_motifs=np.concatenate((noisy_motifs,clean_motifs),axis=0)
	
	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    #ax[shapes2].set_title("PSTH",fontsize=18)
    #py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.145, 0.97, "Syllable A", va="center", ha="left",fontsize=18)
    #py.fig.text(0.39, 0.97, "Syllable B", va="center", ha="left",fontsize=18)
    #py.fig.text(0.72, 0.97, "Syllable C", va="center", ha="left",fontsize=18)
	
    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    #py.fig.text(0.17, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.38, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.73, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_off=noisy_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=noisy_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
               n0+=1  			   
   
           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[0])	 
		   
		   #clean
           spikes1=[]
           n0=0
           used_off=clean_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=clean_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               n0+=1   

           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[0]) 

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[-1])	 
		   
		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           #shift_syl_plot=shift_syl_plot+shoulder_end
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[-1]) 


        elif(i!=2*idx_noisy_syb):
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
		   #noisy  
           used_on=noisy_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  

           #bins=np.arange(0,shoulder_beg, step=binwidth)	  
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])	

		   #clean
           spikes1=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
	
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i]) 

		#treat the spikes within the syllable, that is targeted with noise	   
        else:	
           #noisy renditions	    
           used_on=noisy_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=noisy_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           res=-1
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_noisy[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_noisy=spikes1
                  spikes3_noisy=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_noisy,res+np.zeros(len(spikes3_noisy)),marker="|", color="blue")
                  spikes2_noisy=np.concatenate((spikes2_noisy,shift_syl_plot+spikes3_noisy),axis=0)
                  n0+=1
               else:#syllable not sung
                  res=res+1 
           #bins_edge=bins_edge+meandurall_list[i]		
           #shift_syl_plot=shift_syl_plot+meandurall_list[i]
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           weights_noisy=np.append(weights_noisy,np.ones(len(np.concatenate(spikes1)))/normfactor_noisy[i])
				  
           #clean renditions	  
           used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
           #used_on=used_on[(np.where((used_on >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
           #used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off/fs
           spikes1=[]
           #res=-1 continue with previous value of res
           spikes_cln=[]
           n0=0
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #spikes2_cln=spikes1
                  spikes3_cln=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3_cln,res+np.zeros(len(spikes3_cln)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3_cln),axis=0)
                  n0+=1 
               else:#syllable not sung
                  res=res+1
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i])	

    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    spikes_noisy=np.sort(spikes2_noisy)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])
    y_noisy,x1= py.histogram(spikes_noisy, bins=bins, weights=weights_noisy)#np.ones(len(spikes))/normfactor[0])
	
    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_noisy, color="blue")
    ax[shapes2].plot(x2,y_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
	

    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2)) 
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()
	

	
## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth_glob_sep() but for the case there is no noisy syllables
# If no probabilistic noise feedback, use this function by setting idx_noisy_syb to some irrelevant value (for example -2) that is not caught by the if conditions in the code
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation  
#
#  returns the psth
def psth_glob_sep_no_noise(spikefile, motifile, basebeg, basend,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #len_motif=len(sybs) #length of the motif (nb syllables)

    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    all_motifs=clean_motifs

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
	
    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for clean and noisy motifs
    meandurall_list_clean = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_clean = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    meandurall_list_noisy = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor_noisy = np.zeros(2*len_motif-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):       
		#clean versions of the syll
        used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
        used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
        used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
        #print(used_off)
        #print("\n")
        #print(used_on)
        #print("\n")
        #print("\n")
        used_on=used_on/fs
        used_off=used_off/fs

        meandurall=np.mean(used_off[:]-used_on[:])
        n_clean=len(used_off[:])
        meandurall_list_clean[i]=meandurall
        normfactor_clean[i]=len(used_off[:])
				   
    normfactor_mean_clean=np.mean(normfactor_clean)
    mean_nb_rendit_syl_clean=normfactor_mean_clean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean_clean=normfactor_mean_clean*binwidth
    normfactor_clean=normfactor_clean*binwidth

    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for all motifs
    meandurall_list = np.zeros(2*len_motif-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif-1):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth	
	

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
	#######################################################
	##X ticks: onset and offset of syllables
	#######################################################
    #x_ax_len=shoulder_beg
    #for i in range(2*len_motif-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
    #    x_ticks.append(x_ax_len)
    #    x_ax_len=x_ax_len+meandurall_list[i]
    #x_ticks.append(x_ax_len)
    #x_ticks.append(x_ax_len+shoulder_end)
	#######################################################

	######################################################
	#X ticks: only onset of syllables
	######################################################
    x_ax_len=shoulder_beg
    for i in range(len_motif): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        if(i!=(len_motif-1)):
            x_ax_len=x_ax_len+meandurall_list[2*i]+meandurall_list[2*i+1]
        else:
            x_ax_len=x_ax_len+meandurall_list[2*i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
	######################################################

    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    for b in range(n_baseline):
        baseline_counts_aux=0
        for j in range(mean_nb_rendit_syl):
            basecuts=np.random.choice(np.arange(basebeg,basend))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for len(used) (i.e. the number of syll renditions) random distributions of a bin of size binwidth
    #print(normfactor_mean)
	
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
    ##py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.17, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.46, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.77, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 
	
    for i in range(len_motif):
        py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	
	
	#treat all syllables and gaps
    for i in range(-1,2*len_motif):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_off=clean_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=clean_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               n0+=1   

           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[0]) 

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif-1):
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           #shift_syl_plot=shift_syl_plot+shoulder_end
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[-1]) 


        else:
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
	
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i]) 

		
    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])

    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()

    return 	y_cln


	
## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles. Similar to psth_glob_sep_no_noise() but uses the interval of 1 or 2 seconds before the motif onset as the baseline 
# If no probabilistic noise feedback, use this function by setting idx_noisy_syb to some irrelevant value (for example -2) that is not caught by the if conditions in the code
# The DTW is done for each syllable and gap separately and the mean and std spk for baseline fr is computed once. 
# DTW done separately for clean syllables and for noisy syllables (syll with w noise)
# PSTH done separately for clean motifs and noisy motifs and is superimposed on the final plot
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency 
# 
# n is the index of the BOS (n=0), rBOS (n=1) or OCS (n=2)  
def psth_glob_sep_no_noise_pb(spikefile, motifile,n,binwidth=binwidth, fs=fs):      
    #sybs=["A","B","C"]
    #sybs=["rC","rB","rA"]
    #sybs=["S","T","U","V","W"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    finallist=sortsyls_psth_glob(motifile,n)
    baseline_pb=baseline_playback(motifile)
    clean_motifs=np.array(finallist[0])
    #print(clean_motifs)

    noisy_motifs=np.array(finallist[1])
    all_motifs=clean_motifs
	
    if(n==1 or n==2):# BOS or rBOS
       len_motif_=len_motif
    else:#OCS
       len_motif_=len_OCS
	   
    #print(len_motif_)
    #len_motif=len(sybs) #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder_beg= 0.05 #in seconds
    shoulder_end= 0.05 #in seconds
    meandurall=0
    mean_nb_rendit_syl=0
    meandurall_syl=0#mean of the durations of all syllable types
    normfactor_mean=0
    n_baseline=200
    hist_bin=1
    sig_fr=0 #value of significance of the fr(firing rate) relative to the mean fr 
    last_syl=0
    shift_syl_plot=0
    shapes = (1,)
    shapes2 = (0,)
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,1, figsize=(25,12), sharey=False)

	#Go through the list of syllables and compute the mean duration of each syllable type. 
	#The duration of the syllable that receives probabilistic noise is (mean_dur_syll_clean + mean_dur_syll_noise/2)
	#It is assumed the noise is output after syllable onset and end before syllable offset
	#Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl
	
    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for clean and noisy motifs
    meandurall_list_clean = np.zeros(2*len_motif_-1) #mean duration of each syllable type and gaps
    normfactor_clean = np.zeros(2*len_motif_-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    meandurall_list_noisy = np.zeros(2*len_motif_-1) #mean duration of each syllable type and gaps
    normfactor_noisy = np.zeros(2*len_motif_-1) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    # for i in range(2*len_motif_-1):       
	# 	#clean versions of the syll
    #     used_off=clean_motifs[:,i+1] # sets the offsets of which syllable to use
    #     used_on=clean_motifs[:,i] # sets the onsets of which syllable to use
    #     used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
    #     used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
    #     #print(used_off)
    #     #print("\n")
    #     #print(used_on)
    #     #print("\n")
    #     #print("\n")
    #     used_on=used_on/fs
    #     used_off=used_off/fs
    # 
    #     meandurall=np.mean(used_off[:]-used_on[:])
    #     n_clean=len(used_off[:])
    #     meandurall_list_clean[i]=meandurall
    #     normfactor_clean[i]=len(used_off[:])
	# 			   
    # normfactor_mean_clean=np.mean(normfactor_clean)
    # mean_nb_rendit_syl_clean=normfactor_mean_clean #before *binwidth, normfactor_mean is the number of motif renditions
    # normfactor_mean_clean=normfactor_mean_clean*binwidth
    # normfactor_clean=normfactor_clean*binwidth

    #Compute normfactor_mean(the number of renditions of the motif*binwidth), mean_nb_rendit_syl for all motifs
    meandurall_list = np.zeros(2*len_motif_-1) #mean duration of each syllable type and gaps
    normfactor = np.zeros(2*len_motif_) #number of renditions of each syllable/gap, the noisy syll is the last in the array 
    for i in range(2*len_motif_-1):
           used_off=all_motifs[:,i+1] # sets the offsets of which syllable to use
           used_on=all_motifs[:,i] # sets the onsets of which syllable to use
           used_on=used_on[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_off=used_off[(np.where((used_off >0) == True))] #clean from case where the syllable is not sung
           used_on=used_on/fs
           used_off=used_off/fs

           meandurall=np.mean(used_off[:]-used_on[:])
           normfactor[i]=len(used_off[:])
           meandurall_list[i]=meandurall
           normfactor_mean=len(used_off[:]) #the mean number of motif renditions is the number of renditions of any syllable except the one that receives contingent noise
        
		   
    mean_nb_rendit_syl=normfactor_mean #before *binwidth, normfactor_mean is the number of motif renditions
    normfactor_mean=normfactor_mean*binwidth
    normfactor=normfactor*binwidth	
	

	#Compute the length of the x axis for the plots: shoulder_beg+meandurall_sa+shoulder_end+shoulder_beg+meandurall_sb+shoulder_end+.....
    x_axis_length = 0
    for i in range(2*len_motif_-1):
        x_axis_length=x_axis_length+meandurall_list[i]
    x_axis_length=x_axis_length+shoulder_end+shoulder_beg

	#Set x_axis parameters, ticks, lims, bins
    bins=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes].set_xlim(min(bins), max(bins))
    ax[shapes2].set_xlim(min(bins), max(bins))
    x_ticks=[]
    #x_ticks.append(min(bins))
    x_ax_len=shoulder_beg
    for i in range(2*len_motif_-1): #last element of meandurall_lis is the duration of the noisy version of the syllable receiving contingent noise
        x_ticks.append(x_ax_len)
        x_ax_len=x_ax_len+meandurall_list[i]
    x_ticks.append(x_ax_len)
    x_ticks.append(x_ax_len+shoulder_end)
		
    x_ticks=np.asarray(x_ticks)
    #ax[shapes].set_xticks([min(bins),0,meandurall_list[i],max(bins)])
    ax[shapes].set_xticks(x_ticks)
    ax[shapes2].set_xticks(x_ticks)
    
	#################################
    # Computation of baseline
	#################################
    baseline_counts=[] 
    len_baseline_pb=len(baseline_pb[0]) #nb of baseline chuncks
    baseline_width=0.2
    #print(len_baseline_pb)
    for b in range(n_baseline):
        baseline_counts_aux=0
        #print(b)
        for j in range(mean_nb_rendit_syl):
            idx_baseline=np.random.choice(np.arange(0,len_baseline_pb)) #Take randomly a baseline chunk
            basebeg_pb=baseline_pb[0][idx_baseline]/fs
            basend_pb=baseline_pb[1][idx_baseline]/fs
            basecuts=np.random.choice(np.arange(basebeg_pb,basend_pb,baseline_width))
            baseline_counts_aux+=len(spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+binwidth) == True)]) #add number of spikes in randomly distributed bin
        baseline_counts+=[baseline_counts_aux/normfactor_mean] #mean value of the fr computed for lmean_nb_rendit_syl (i.e. the number of syll renditions) random distributions of a bin of size binwidth

    #print(baseline_counts)
    basemean=np.mean(baseline_counts) 
    stdbase=np.ceil(np.std(baseline_counts))
    hist_width=(int)(stdbase*10)
    baseline_counts=baseline_counts-basemean
    bins_base=np.arange(-hist_width,hist_width+1,hist_bin)
    u,_=py.histogram(baseline_counts, bins_base,density=True)
    #py.figure()
    #py.plot(u)
    #compute the significance level for fr beyond basemean
    cumul_sig=0
    mid_hist=(int)(hist_width/hist_bin)
    #determine the level of significance for the fr (sig_fr)
    #start from the middle of the histogram and go to the edges on both sides and count the cummulated area under the histogram till threshold of 95%
    for j in range(hist_width):
        cumul_sig=cumul_sig+u[mid_hist+j]*hist_bin+u[mid_hist-j]*hist_bin
        if(cumul_sig >= 0.95):
           break
    	
    sig_fr=j*hist_bin
    #print(sig_fr)	

	##############################
	#Set axis for plot
	##############################
    #axis=np.arange(meandurall_list[0]/3,meandurall_list[0]*2/3,binwidth)
    axis=np.arange(x_axis_length+binwidth, step=binwidth)
    ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+sig_fr), color = "black")
    ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-sig_fr), color = "black", ls="dashed")
	
	
    if (n==1): #BOS 
       for i in range(len_motif_):
           py.fig.text(pos_syls_PSTH[i], 0.97, "Syllable " + sybs[i], va="center", ha="left",fontsize=18)      
    elif(n==2): #rBOS
       for i in range(len_motif_):
           py.fig.text(pos_syls_rBOS_PSTH[i], 0.97, "Syllable " + "r" + sybs[len_motif_-i-1], va="center", ha="left",fontsize=18)      
    else: #OCS
       for i in range(len_motif_):
           py.fig.text(pos_syls_OCS_PSTH[i], 0.97, "Syllable " + sybs_OCS[i], va="center", ha="left",fontsize=18)      
	
	
    ###ax[shapes2].set_title("PSTH",fontsize=18)
	##rBOS
    ##py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.24, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.55, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.76, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18)   

	##BOS
    ##py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.16, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.38, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.67, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 
	
	
	##OCS
    ##py.fig.text(0.5, 1, "PSTH", va="center", ha="center",fontsize=20)
    #py.fig.text(0.08, 0.97, "Syllable " + sybs[0], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.25, 0.97, "Syllable " + sybs[1], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.42, 0.97, "Syllable " + sybs[2], va="center", ha="left",fontsize=18) 
    #py.fig.text(0.57, 0.97, "Syllable " + sybs[3], va="center", ha="left",fontsize=18)      
    #py.fig.text(0.78, 0.97, "Syllable " + sybs[4], va="center", ha="left",fontsize=18) 	
	
	

    ax[shapes2].tick_params(
            axis="x",          # changes apply to the x-axis
            which="both",      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)	
    ax[shapes2].tick_params(axis='both', which='major', labelsize=18)
    ax[shapes].tick_params(axis='both', which='major', labelsize=18)
	
    bins_edge=0
    spikes2_cln=np.array([])
    spikes2_noisy=np.array([])
    weights_cln=np.array([],dtype=float)
    weights_noisy=np.array([],dtype=float)
	
	
	#treat all syllables and gaps
    for i in range(-1,2*len_motif_):
	    #treat the spikes in the shoulder window before motif onset
        if(i==-1):
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_off=clean_motifs[:,0] #sets the onsets of firstt sylable in motif
           used_off=used_off/fs
           used_on=clean_motifs[:,0] 
           used_on=(used_on/fs)-shoulder_beg #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
               spikes1+=[step1]
               res=res+1 #motif numer shift on y axis on raster plot
               #spikes2=spikes1
               spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	   #Shift the spike times for each syllable type for the scatter plot 
               ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
               spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
               n0+=1   

           bins_edge=bins_edge+shoulder_beg		
           shift_syl_plot=shift_syl_plot+shoulder_beg
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[0]) 

	    #treat the spikes in the shoulder window after motif offset
        elif(i==2*len_motif_-1):
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i] 
           used_off=(used_off/fs)+shoulder_end #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0): #last syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  spikes1+=[step1]
                  res=res+1 #motif numer shift on y axis on raster plot
                  #spikes2=spikes1
                  spikes3=np.array(spikes1[n0]) # Gets the step1 array for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1
               else: #last syllable not sung
                  res=res+1
           bins_edge=bins_edge+shoulder_end	
           #shift_syl_plot=shift_syl_plot+shoulder_end
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[-1]) 


        else:
		   #clean
           spikes1=[]
           res=-1
           spikes=[]
           n0=0
           used_on=clean_motifs[:,i] #sets the onsets of firstt sylable in motif
           used_on=used_on/fs
           used_off=clean_motifs[:,i+1] 
           used_off=used_off/fs #considers the time delay due to shoulder beg
           for j in range(len(used_on)):
               step1=[]
               step2=[]
               step3=[]
               beg= used_on[j] #Will compute the beginning of the window
               end= used_off[j] #Will compute the end of the window
               if(beg>0 and end>0):#syllable sung
                  step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]-beg
                  step2=step1*(meandurall_list_clean[i]/(end-beg))
                  spikes1+=[step2]
                  res=res+1
                  #print(step2)
                  spikes3=np.array(spikes1[n0]) # Gets the step2 and step3 arrays for scatter
		   	      #Shift the spike times for each syllable type for the scatter plot 
                  ax[shapes].scatter(shift_syl_plot+spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
                  spikes2_cln=np.concatenate((spikes2_cln,shift_syl_plot+spikes3),axis=0)
                  n0+=1		   
               else:#syllable not sung
                  res=res+1  
	
           #ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           #ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           ax[shapes].axvline(x=shift_syl_plot, color="grey", linestyle="--")
           ax[shapes2].axvline(x=shift_syl_plot, color="grey", linestyle="--")	
           bins_edge=bins_edge+meandurall_list[i]		
           shift_syl_plot=shift_syl_plot+meandurall_list[i]
           weights_cln=np.append(weights_cln,np.ones(len(np.concatenate(spikes1)))/normfactor_clean[i]) 

		
    #########################
    # Computation of spikes
	#########################
    bins=np.arange(0,bins_edge, step=binwidth)
    #spikes=np.sort(np.concatenate(spikes2))
    spikes_cln=np.sort(spikes2_cln)
    y_cln,x1= py.histogram(spikes_cln, bins=bins, weights=weights_cln)#np.ones(len(spikes))/normfactor[0])

    #if np.mean(y1) < 5:
    #    f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
	
	#set new x axis by shifting the bin edges by binwidth/2
    x2=np.delete(x1,-1)
    x2=x2+binwidth/2
    ax[shapes2].plot(x2,y_cln, color="red")
    #ax[shapes2].plot(xnew,ynew, color="green")
    py.fig.subplots_adjust(hspace=0)	

    #ax[shapes2].plot(xnew,ynew, color="blue")
    py.fig.subplots_adjust(hspace=0)
    black_line = mlines.Line2D([], [], color="black", label="+95%")
    black_dashed  = mlines.Line2D([], [], color="black", label="-95%", linestyle="--")
    green_line  = mlines.Line2D([], [], color="green", label="Mean")
    ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 12})
    ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	#leg = ax[0].legend(loc="upper left", bbox_to_anchor=[0, 1], fancybox=True)
		
    if (len_motif_ == 1):
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
    else:
        ax[0].set_ylabel("Spikes/Sec",fontsize=18)
        ax[1].set_ylabel("Motif number",fontsize=18)
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        values = np.array(ax[0].get_ylim())
        values2 = np.array(ax[1].get_ylim())
        top = np.sort(np.append(top, values))
        top2 = np.sort(np.append(top2, values2))
        ax[0].set_ylim(0,max(top))
        ax[1].set_ylim(min(top2),max(top2))  
        #for lim in range(len_motif_):
        #    values = np.array(ax[0,lim].get_ylim())
        #    values2 = np.array(ax[1,lim].get_ylim())
        #    top = np.sort(np.append(top, values))
        #    top2 = np.sort(np.append(top2, values2))
        #for limreal in range(len(finallist)):
        #    ax[0,limreal].set_ylim(0,max(top))
        #    ax[1,limreal].set_ylim(min(top2),max(top2))   

    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.042, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.03, "Time(seconds)", va="center", ha="center",fontsize=18)
    f.close()
	
	

## 
#
# This function can be used to obtain the cutting points of the syllables for corrpitch_auto, corramplitude_auto and corrspectral_auto entrop correlations
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate.
#
def syl_cut(songfile, motifile, fs=fs, window_size=window_size):
    
    #Read and import files that will be needed
    song=np.load(songfile)
    finallist=sortsyls(motifile,0)  
       
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=finallist[0]
    elif answer.lower() == "b":
        used=finallist[1]
    elif answer.lower() == "c":
        used=finallist[2]    
    elif answer.lower() == "d":
        used=finallist[3]
    elif answer.lower() == "e":
        used=finallist[4]
    elif answer.lower() == "f":
        used=finallist[5]
    elif answer.lower() == "g":
        used=finallist[6]
    
    #Will plot an exmaple of the syllable for you to get an idea of the number of chunks
    fig, az = py.subplots()
    example=song[int(used[0][0]):int(used[0][1])]
    tempo=np.linspace(used[0][0]/fs, used[0][1]/fs, len(example))
    abso=abs(example)
    az.plot(tempo,example)
    az.plot(tempo,abso)
    smooth=smoothed(np.ravel(example),fs)
    az.plot(tempo[:len(smooth)],smooth)
    az.set_title("Click on graph to move on.")
    py.waitforbuttonpress(10)
    numcuts=int(input("Number of chunks?"))
    py.close()
    
    # Will provide you 10 random exmaples of syllables to stablish the cutting points
    coords2=[]
    for j in range(10):           
       j=random.randint(0,len(used)-1)
       fig, ax = py.subplots()
       syb=song[int(used[j][0]):int(used[j][1])]
       
       #original by Eduarda
       #abso=abs(syb)
       #ax.plot(abso)
       #rms=window_rms(np.ravel(syb),window_size)
       #ax.plot(rms)
       
       syb=syb[:,0]
       syb_smth=bandpass_filtfilt(syb, fs) #high pass( 500Hz) filter syllable
       abso_smth=abs(syb_smth)
       ax.plot(abso_smth)
       rms_smth=window_rms(np.ravel(syb_smth),window_size)
       ax.plot(rms_smth)
    
       py.waitforbuttonpress(10)
       while True:
           coords = []
           while len(coords) < numcuts+1:
               tellme("Select the points to cut with mouse")
               coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
           scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
           tellme("Happy? Key click for yes, mouse click for no")
           if py.waitforbuttonpress():
               break
           else:
               scat.remove()
       py.close()
       coords2=np.append(coords2,coords[:,0])
    
    #Will keep the mean coordinates for the cuts
    coords2.sort()
    coords2=np.split(coords2,numcuts+1)
    means=[]
    for k in range(len(coords2)):
        means+=[int(np.mean(coords2[k]))]
    np.savetxt("Mean"+answer+".txt", means) 
    
    # Will plot how the syllables will be cut according to the avarage of the coordinates clicked before by the user
    py.plot(syb_smth)
    for l in range(1,len(means)):
        py.plot(np.arange(means[l-1],means[l-1]+len(syb_smth[means[l-1]:means[l]])),syb_smth[means[l-1]:means[l]])   


## 
#
# This function can be used to obtain the FWHM (Full Width at Half Maximum) of an LFP spike
#		
def FWHM_measurement():
    #Baseline
    #LFP
    os.chdir("Files_spk_shapes")
    file_spike_shapes="LFPShape_baseline_plot#ch16#1.txt" #"LFPShape_baseline_plot#ch16#1.txt"
    #file_spike_shapes = current_dir+'\\'+songfile
    fp=open(file_spike_shapes,"r")
    lines = fp.readlines()
    samples_LFP=[]
    samples_notLFP=[]
    fig, s = py.subplots(2, 2)
    i=0
    for line_p in lines[1:]:
        #print(line_p)
        splt_p = line_p.split()
        LFP_=np.array(splt_p[2:],dtype=float)
        samples_LFP+=[LFP_]
        s[0,0].plot(LFP_,color="black")
        #s[1,0].plot(notLFP_,color="black")
        i=i+1
    			   		   				   
    fp.close
    samples_LFP=np.array(samples_LFP)
    #print(samples_LFP.shape)
    mean_samples_LFP=np.mean(samples_LFP,axis=0)
    s[0,1].plot(np.mean(samples_LFP,axis=0),color="black")


    #Spike
    file_spike_shapes="SpikeShape_baseline_plot#ch16#1.txt"
    #file_spike_shapes = current_dir+'\\'+songfile
    fp=open(file_spike_shapes,"r")
    lines = fp.readlines()
    samples_LFP=[]
    samples_notLFP=[]
    #fig, s = plt.subplots(2, 2)
    for line_p in lines[1:]:
        #print(line_p)
        splt_p = line_p.split()
        LFP_=np.array(splt_p[2:],dtype=float)
        samples_LFP+=[LFP_]
        s[1,0].plot(LFP_,color="black")
        #s[1,0].plot(notLFP_,color="black")
    			   		   				   
    fp.close
    samples_LFP=np.array(samples_LFP)
    #samples_notLFP=np.array(samples_notLFP)
    s[1,1].plot(np.mean(samples_LFP,axis=0),color="black")
    
    s[0,1].set_title("Mean SpikeShape LFP base")
    s[1,1].set_title("Mean SpikeShape filtered base") # Just like you would see in Spike2
    s[0,1].set_ylabel("Amplitude")
    s[1,1].set_ylabel("Amplitude")
    s[1,1].set_xlabel("Sample points")
    #py.tight_layout()
    s[0,0].set_title("SpikeShape LFP base")
    s[1,0].set_title("SpikeShape filtered base") # Just like you would see in Spike2
    s[0,0].set_ylabel("Amplitude")
    s[1,0].set_ylabel("Amplitude")
    s[1,0].set_xlabel("Sample points")


	#Ask the user to select points for spike FWHM measurement
    fig, ax = py.subplots()
    ax.plot(mean_samples_LFP)
    
    py.waitforbuttonpress(10)
    while True:
        coords = []
        while len(coords) < 2:
            tellme("Select the points to cut with mouse")
            coords = np.asarray(py.ginput(2, timeout=-1, show_clicks=True))
        scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
        tellme("Happy? Key click for yes, mouse click for no")
        if py.waitforbuttonpress():
            break
        else:
            scat.remove()
    py.close()
    print(coords[0,0]) # x-coords first point (peak)
    print(coords[0,1]) # y-coords first point (peak)
    print(coords[1,0]) # x-coords second point (baseline)
    print(coords[1,1]) # y-coords second point (baseline)
	
	#Maximum of the spike:
    max_spike = abs(coords[0,1]-coords[1,1])
    threshold=min(coords[0,1],coords[1,1])+max_spike/2.0
    print(threshold)
    print((len(mean_samples_LFP)))
    
    fig, ax = py.subplots()
    ax.plot(mean_samples_LFP)
    #print(threshold*np.ones(len(mean_samples_LFP)))
    ax.plot(threshold*np.ones(len(mean_samples_LFP)),'r')
    above_spk=np.where((mean_samples_LFP >= threshold) == True)
    print(above_spk)
    print(max(above_spk[0]))
    spike_dur=max(above_spk[0])-min(above_spk[0])+1
    print(spike_dur)
    np.savetxt("LFP_spk_duration_samplunit.txt", [spike_dur]) 
    os.chdir("..")
    #coords2=np.append(coords2,coords[:,0])

## 
#
# Generates correlations for each syllable. 
# Deprecated: no correlation plot outputs, asks how to cut the syllable....
#
# Arguments:
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# n_iterations is the number of iterations for the bootstrapping
#
# fs is the sampling frequency
def corrduration(spikefile, motifile, n_iterations=n_iterations,fs=fs):      
    #Read and import mat file (new version)
    sybs=["A","B","C","D","E"]
    finallist=sortsyls(motifile,0)    
    #Starts to compute correlations and save the data into txt file (in case the user wants to use it in another software)
    spused=np.loadtxt(spikefile)
    check=jumpsyl(spikefile)
    final=[]
    f = open("SummaryDuration.txt", "w+")
    for i in range(len(finallist)):
        if sybs[i] in check:
            continue
        else:
            used=finallist[i]/fs
            dur=used[:,1]-used[:,0]
            array=np.empty((1,2))
            statistics=[]
            for j in range(len(used)):
                step1=[]
                beg= used[j][0] #Will compute the beginning of the window
                end= used[j][1] #Will compute the end of the window
                step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                array=np.append(array, np.array([[dur[j]],[np.size(step1)/dur[j]]]).reshape(-1,2), axis=0)
            array=array[1:]
            np.savetxt("Data_Raw_Corr_Duration_Result_Syb"+str(sybs[i])+".txt", array, header="First column is the duration value, second is the number of spikes.")
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            z = np.abs(scipy.stats.zscore(array))
            array=array[(z < threshold).all(axis=1)]
            if len(array)<3:
                continue
            else:               
                s1=scipy.stats.shapiro(array[:,0])[1]
                s2=scipy.stats.shapiro(array[:,1])[1]
                s3=np.array([s1,s2])
                s3=s3>alpha
                homo=scipy.stats.levene(array[:,0],array[:,1])[1]
                if  s3.all() == True and homo > alpha: #test for normality
                    final=scipy.stats.pearsonr(array[:,0],array[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(array[:,0],array[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for x in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_Duration_Result_Syb"+str(sybs[i])+".txt", np.array(statistics), header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.") #First column is the correlation value, second is the p value.
                print("Syllable " + str(sybs[i]) +": " + str(final))
                f.writelines("Syllable " + str(sybs[i]) +": " + str(final) + "\n")
   


## 
#
# Generates correlations for each syllable. 
# To be used it with new matfiles. Based on corrduration but figures and results are placed in separate folders
#
# Arguments:
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# n_iterations is the number of iterations for the bootstrapping
#
# fs is the sampling frequency
def corrduration_auto(spikefile, motifile, n_iterations,fs):
    print(os.getcwd())      
    #Read and import mat file (new version)
    Syls=["a","b","c","d","e","g"] 
    finallist=sortsyls(motifile,0)    
    #Starts to compute correlations and save the data into txt file (in case the user wants to use it in another software)
    spused=np.loadtxt(spikefile)
    check=jumpsyl(spikefile)
    final=[]
    f = open("SummaryDuration.txt", "w+")
    for i in range(len(finallist)):
        if Syls[i] in check:
            continue
        else:        
            used=finallist[i]/fs
            dur=used[:,1]-used[:,0]
            #print(dur)
            array=np.empty((1,2))
            statistics=[]
            for j in range(len(used)):
                step1=[]
                beg= used[j][0] #Will compute the beginning of the window
                end= used[j][1] #Will compute the end of the window
                step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                array=np.append(array, np.array([[dur[j]],[np.size(step1)/dur[j]]]).reshape(-1,2), axis=0)
            array=array[1:]
            os.chdir("Results")
            np.savetxt("Data_Raw_Corr_Duration_Result_Syb"+str(Syls[i])+".txt", array, header="First column is the duration value, second is the number of spikes.")
            os.chdir("..")
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            z = np.abs(scipy.stats.zscore(array))
            array=array[(z < threshold).all(axis=1)]
            if len(array)<3:
                continue
            else:               
                s1=scipy.stats.shapiro(array[:,0])[1]
                s2=scipy.stats.shapiro(array[:,1])[1]
                s3=np.array([s1,s2])
                s3=s3>alpha
                homo=scipy.stats.levene(array[:,0],array[:,1])[1]
                if  s3.all() == True and homo > alpha: #test for normality
                    final=scipy.stats.pearsonr(array[:,0],array[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]] #correlation coeff, p-value
                    if(final[1]<alpha):
                       # Create linear regression object
                       regr = LinearRegression()
                       # Train the model using the training sets
                       x_dur=(array[:,0]).reshape(-1,1)
                       y_fr=(array[:,1]).reshape(-1,1)
                       fr_mean=np.mean(y_fr)
                       y_fr=y_fr-fr_mean
                       regr.fit(x_dur, y_fr)
                       # Make predictions using the testing set
                       fr_pred = regr.predict(x_dur)
                       py.fig, ax = py.subplots(1, figsize=(25,12))
                       ax.plot(x_dur,y_fr,'bo')
                       ax.plot(x_dur,fr_pred,'r')
                       ax.set_xlabel("Duration syllable "+Syls[i] +" (seconds)",fontsize=18)
                       ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                       ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                       ax.tick_params(
                           axis="both",          # changes apply to the x-axis
                           which="major",      # both major and minor ticks are affected
                           labelsize=16)	
                       wind=py.get_current_fig_manager()
                       wind.window.showMaximized()
					   #save above figure
                       os.chdir("Figures")
                       py.savefig("Corr_Duration_syb"+ Syls[i] +"_scatter.jpg")
                       py.close()
                       os.chdir("..")

                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(array[:,0],array[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]] #correlation coeff, p-value
                    if(final[1]<alpha):
                       # Create linear regression object
                       regr = LinearRegression()
                       # Train the model using the training sets
                       x_dur=(array[:,0]).reshape(-1,1)
                       y_fr=(array[:,1]).reshape(-1,1)
                       fr_mean=np.mean(y_fr)
                       y_fr=y_fr-fr_mean
                       regr.fit(x_dur, y_fr)
                       # Make predictions using the testing set
                       fr_pred = regr.predict(x_dur)
                       py.fig, ax = py.subplots(1, figsize=(25,12))
                       ax.plot(x_dur,y_fr,'bo')
                       ax.plot(x_dur,fr_pred,'r')
                       ax.set_xlabel("Duration syllable "+Syls[i] +" (seconds)",fontsize=18)
                       ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                       ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                       ax.tick_params(
                           axis="both",          # changes apply to the x-axis
                           which="major",      # both major and minor ticks are affected
                           labelsize=16)	
                       wind=py.get_current_fig_manager()
                       wind.window.showMaximized()	
					   #save above figure
                       os.chdir("Figures")
                       py.savefig("Corr_Duration_syb"+ Syls[i] +"_scatter.jpg")
                       py.close()
                       os.chdir("..")
					   
                    # Bootstrapping
                    for x in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                os.chdir("Results")
                np.savetxt("Data_Boot_Corr_Duration_Result_Syb"+str(Syls[i])+".txt", np.array(statistics), header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.") #First column is the correlation value, second is the p value.
                os.chdir("..")
                print("Syllable " + str(Syls[i]) +": " + str(final))
                f.writelines("Syllable " + str(Syls[i]) +": " + str(final) + "\n")
        

## 
#
# This function allows you to see the envelope for song signal.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# window_size is the size of the window for the convolve function. 
def plotEnvelopes(songfile, beg, end, window_size=window_size): 
    inputSignal=np.load(songfile)
    inputSignal=np.ravel(inputSignal[beg:end])
    
    outputSignal=getEnvelope(inputSignal, window_size)
    rms=window_rms(inputSignal, window_size)
    
    # Plots of the envelopes
    py.fig, (a,b,c) =py.subplots(3,1, sharey=True)
    py.xlabel("Sample Points")
    a.plot(abs(inputSignal))
    a.set_ylabel("Amplitude")
    a.set_title("Raw Signal")
    b.plot(abs(inputSignal))
    b.plot(outputSignal)
    b.set_ylabel("Amplitude")
    b.set_title("Squared Windows")
    c.plot(abs(inputSignal))
    c.plot(rms)           
    c.set_ylabel("Amplitude")
    c.set_title("RMS")
    py.tight_layout()
    py.show()

## 
#
# This function will perform the Fast Fourier Transform to obtain the power spectrum of the syllables.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# fs is the sampling rate
def powerspectrum(songfile, beg, end, fs=fs):
    signal=np.load(songfile) #The song channel raw data
    signal=signal[beg:end] #I selected just one syllable A to test
    print ("Frequency sampling", fs)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs)
    print ("secs", secs)
    Ts = 1.0/fs # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fft(signal))**2 # if **2 is power spectrum, without is amplitude spectrum
    FFT_side = FFT[range(int(N/2))] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(int(N/2))]
    py.subplot(311)
    py.plot(t, signal, "g") # plotting the signal
    py.xlabel("Time")
    py.ylabel("Amplitude")
    py.subplot(312)
    py.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    py.xlabel("Frequency (Hz)")
    py.title("Double-sided")
    py.ylabel("Power")
    py.subplot(313)
    py.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
    py.xlabel("Frequency (Hz)")
    py.title("Single sided")
    py.ylabel("Power")
    py.tight_layout()
    py.show()

## 
#
# This function can be used to obtain the pitch of specific tones inside a syllable.
# It will execute an autocorrelation for the identification of the pitchs
# Deprecated: no correlation plot outputs, asks how to cut the syllable....
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# lags is the number of lags for the autocorrelation
#
# window_size is the size of the window for the convolve function (RMS of signal)
#
# fs is the sampling rate   
def corrpitch(songfile, motifile,spikefile, lags=lags, window_size=window_size,fs=fs, means=None):
    
    #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile,0)
    
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=finallist[0]
    elif answer.lower() == "b":
        used=finallist[1]
    elif answer.lower() == "c":
        used=finallist[2]  
    elif answer.lower() == "d":
        used=finallist[3]
    
    if means is not None:
        means = np.loadtxt(means).astype(int)
        syb=song[int(used[0][0]):int(used[0][1])]
        pass
    else: 
        #Will plot an exmaple of the syllable for you to get an idea of the number of chunks
        fig, az = py.subplots()
        example=song[int(used[0][0]):int(used[0][1])]
        tempo=np.linspace(used[0][0]/fs, used[0][1]/fs, len(example))
        abso=abs(example)
        az.plot(tempo,example)
        az.plot(tempo,abso)
        rms=window_rms(np.ravel(example),window_size)
        az.plot(tempo[:len(rms)],rms)
        az.set_title("Click on graph to move on.")
        py.waitforbuttonpress(10)
        numcuts=int(input("Number of chunks?"))
        py.close()
        
        # Will provide you 4 random exmaples of syllables to stablish the cutting points
        coords2=[]
        for j in range(4):           
           j=random.randint(0,len(used)-1)
           fig, ax = py.subplots()
           syb=song[int(used[j][0]):int(used[j][1])]
		   
		   #original by Eduarda
           #abso=abs(syb)
           #ax.plot(abso)
           #rms=window_rms(np.ravel(syb),window_size)
		   #ax.plot(rms)
		   
           syb=syb[:,0]
           syb_smth=bandpass_filtfilt(syb, fs) #high pass( 500Hz) filter syllable
           abso_smth=abs(syb_smth)
           ax.plot(abso_smth)
           rms_smth=window_rms(np.ravel(syb_smth),window_size)
           ax.plot(rms_smth)
		   
           py.waitforbuttonpress(10)
           while True:
               coords = []
               while len(coords) < numcuts+1:
                   tellme("Select the points to cut with mouse")
                   coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
               scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
               tellme("Happy? Key click for yes, mouse click for no")
               if py.waitforbuttonpress():
                   break
               else:
                   scat.remove()
           py.close()
           coords2=np.append(coords2,coords[:,0])
        
        #Will keep the mean coordinates for the cuts
        coords2.sort()
        coords2=np.split(coords2,numcuts+1)
        means=[]
        for k in range(len(coords2)):
            means+=[int(np.mean(coords2[k]))]
        np.savetxt("Mean"+answer+".txt", means) 
    
    # Will plot how the syllables will be cut according to the avarage of the coordinates clicked before by the user    
    py.plot(syb_smth)
    for l in range(1,len(means)):
        py.plot(np.arange(means[l-1],means[l-1]+len(syb_smth[means[l-1]:means[l]])),syb_smth[means[l-1]:means[l]])   

    # Autocorrelation and Distribution 
    for m in range(1,len(means)):
        spikespremot=[]
        spikesdur=[]
        freq2=[]
        coords5=[]
        fig=py.figure(figsize=(25,12))
        gs=py.GridSpec(2,2)
        a2=fig.add_subplot(gs[0,0]) # First row, first column
        a3=fig.add_subplot(gs[0,1]) # First row, second column
        a1=fig.add_subplot(gs[1,:]) 
        for n in range(len(used)):
            syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
            sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
            x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
            f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags), unbiased=True), kind="quadratic")
            xnew=np.linspace(min(x2),max(x2), num=1000)
            a1.plot(xnew,f(xnew))
            a1.set_xlabel("Number of Lags")
            a1.set_ylabel("Autocorrelation score")
        a1.set_label(tellme("Want to keep it? Key click (x2) for yes, mouse click for no"))
        gs.tight_layout(fig)
        if not py.waitforbuttonpress(30):
            py.close()
            continue            
        else:
            py.waitforbuttonpress(30)
            while True:           
                coord=[]
                while len(coord) < 2:
                    tellme("Select the points for the peak.") #You should choose in the graph the range that representes the peak
                    coord = np.asarray(py.ginput(2, timeout=-1, show_clicks=True))
                scat=a1.scatter(coord[:,0],coord[:,1], s=50, marker="X", zorder=10, c="b")
                tellme("Happy? Key click for yes, mouse click for no")
                if py.waitforbuttonpress(30):
                    break
                else:
                    scat.remove()
            coords5=coord[:,0]*10 # times ten is because of the linspace being 1000
            a1.clear()
            
        #From now it will use the coordinates of the peak to plot the distribution and the interpolated version of the peak    
        for x in range(len(used)):
            syb=song[int(used[x][0]):int(used[x][1])]
            sybcut=syb[means[m-1]:means[m]]
            x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
            f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags), unbiased=True), kind="quadratic")
            xnew=np.linspace(min(x2),max(x2), num=1000)
            a1.plot(xnew,f(xnew))
            x3=xnew[int(coords5[0]):int(coords5[1])]
            g=scipy.interpolate.interp1d(x3,f(xnew)[int(coords5[0]):int(coords5[1])], kind="cubic")
            xnew2=np.linspace(min(x3),max(x3), num=1000)
            a2.plot(xnew2,g(xnew2))
            peak=np.argmax(g(xnew2))
            freq2+=[xnew2[peak]]
            beg=(used[x][0] + means[m-1])/fs
            end=(used[x][0] + means[m])/fs
            step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
            step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
            spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
            spikesdur+=[[np.size(step2)/(end-beg)]]
        statistics=[]
        statistics2=[]
        spikesdur=np.array(spikesdur)[:,0]
        spikespremot=np.array(spikespremot)[:,0]
        freq2=np.array(freq2)
        freq2=np.reciprocal(freq2/fs)
        total = np.column_stack((freq2,spikespremot,spikesdur))
        np.savetxt("Data_Raw_Corr_Pitch_Result_Syb" + answer + "_tone_" + str(m) + ".txt", total, header="First column is the pitch value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
        #Here it will give you the possibility of computing the correlations and Bootstrapping
        an=input("Correlations?")
        if an.lower() == "n":
            pass
        else:
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            total1=np.column_stack((freq2,spikespremot))
            total2=np.column_stack((freq2,spikesdur))
            z1 = np.abs(scipy.stats.zscore(total1))
            z2 = np.abs(scipy.stats.zscore(total2))
            total1=total1[(z1 < threshold).all(axis=1)]
            total2=total2[(z2 < threshold).all(axis=1)]
            a = total1[:,1] == 0
            b = total2[:,1] == 0
            #This will get the data for Pitch vs Premotor
            if len(total1) < 3 or all(a) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total1[:,0])[1] #Pitch column
                s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                homo=scipy.stats.levene(total1[:,0],total1[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + answer + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                print(final)
            #This will get the data for Pitch vs During     
            if len(total2) < 3 or all(b) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total2[:,0])[1] #Pitch column
                s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + answer + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")    
                print(final)                  
        a2.set_xlabel("Number of Lags")
        a2.set_ylabel("Autocorrelation score")
        a3.hist(freq2, bins=int(np.mean(freq2)*0.01))
        a3.set_xlabel("Frequency (Hz)")
        a1.set_xlabel("Number of Lags")
        a1.set_ylabel("Autocorrelation score")
        a1.set_label(tellme("Now let's select the frequency. Key click (x2) for yes, mouse click for no")) #Here you will be asked to select a point in the peak that could represent the frequency (just to get an estimation)
        gs.tight_layout(fig)
        if not py.waitforbuttonpress(30):
            py.savefig("Corr_Pitch_syb"+ answer +"_tone"+ str(m)+".tif")
            py.close()
            continue            
        else:
            py.waitforbuttonpress(30)
            while True:
                freq = []
                while len(freq) < 1:
                    tellme("Select the point for the frequency.")
                    freq = np.asarray(py.ginput(1, timeout=-1, show_clicks=True))
                scat= a1.scatter(freq[:,0],freq[:,1], s=50, marker="X", zorder=10, c="b") 
                ann=a1.annotate(str(int(np.reciprocal(freq[:,0]/fs))) +" Hz", xy=(freq[:,0],freq[:,1]), xytext=(freq[:,0]*1.2,freq[:,1]*1.2),
                            arrowprops=dict(facecolor="black", shrink=0.05))
                            
                tellme("Happy? Key click for yes, mouse click for no")
                if py.waitforbuttonpress(30):
                    py.savefig("Corr_Pitch_syb"+ answer +"_tone"+ str(m)+".tif")
                    break
                else:
                    ann.remove()
                    scat.remove()


## 
#
# This function can be used to obtain the pitch of specific tones inside a syllable.
# It will execute an autocorrelation for the identification of the pitches. Based on corrpitch but without asking for syllable 
# chunks used for pitch computation. Read edges of the chunks from files
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# lags is the number of lags for the autocorrelation
#
# window_size is the size of the window for the convolve function (RMS of signal)
#
# fs is the sampling rate   
def corrpitch_auto(songfile, motifile, spikefile,lags, window_size=window_size,fs=fs):
    
   #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile,0) 	
    fichier = open("SummaryCorrPitch.txt", "w+")
    y=["Mean_p_A.txt","Mean_p_B.txt","Mean_p_C.txt","Mean_p_D.txt"]
    Syls=["a","b","c","d"]
    check=jumpsyl(spikefile)
    for obj in range(len(finallist)):
        if Syls[obj] in check:
            continue
        else:                
            used=finallist[obj]
            means = np.loadtxt(y[obj]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
        
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                freq2=[]
                coords5=[]
                fig=py.figure(figsize=(25,12))
                gs=py.GridSpec(2,2)
                a2=fig.add_subplot(gs[0,0]) # First row, first column
                a3=fig.add_subplot(gs[0,1]) # First row, second column
                a1=fig.add_subplot(gs[1,:]) 
                fig.suptitle("Syllable " + Syls[obj] + " Tone " + str(m))
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
                    f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags)), kind="quadratic")
                    xnew=np.linspace(min(x2),max(x2), num=1000)
                    a1.plot(xnew,f(xnew))
                    a1.set_xlabel("Number of Lags")
                    a1.set_ylabel("Autocorrelation score")
                a1.set_label(tellme("Want to keep it? Key click (x2) for yes, mouse click for no"))
                if not py.waitforbuttonpress(30):
                    py.close()
                    continue            
                else:
                    py.waitforbuttonpress(30)
                    while True:           
                        coord=[]
                        while len(coord) < 2:
                            tellme("Select the points for the peak.") #You should choose in the graph the range that representes the peak
                            coord = np.asarray(py.ginput(2, timeout=-1, show_clicks=True))
                        scat=a1.scatter(coord[:,0],coord[:,1], s=50, marker="X", zorder=10, c="b")
                        tellme("Happy? Key click for yes, mouse click for no")
                        if py.waitforbuttonpress(30):
                            break
                        else:
                            scat.remove()
                    coords5=coord[:,0]*10 # times ten is because of the linspace being 1000
                    a1.clear()
                    
                #From now it will use the coordinates of the peak to plot the distribution and the interpolated version of the peak    
                for x in range(len(used)):
                    syb=song[int(used[x][0]):int(used[x][1])]
                    sybcut=syb[means[m-1]:means[m]]
                    x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
                    f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags)), kind="quadratic")
                    xnew=np.linspace(min(x2),max(x2), num=1000)
                    a1.plot(xnew,f(xnew))
                    x3=xnew[int(coords5[0]):int(coords5[1])]
                    g=scipy.interpolate.interp1d(x3,f(xnew)[int(coords5[0]):int(coords5[1])], kind="cubic")
                    xnew2=np.linspace(min(x3),max(x3), num=1000)
                    a2.plot(xnew2,g(xnew2))
                    peak=np.argmax(g(xnew2))
                    freq2+=[xnew2[peak]]
                    beg=(used[x][0] + means[m-1])/fs
                    end=(used[x][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                statistics=[]
                statistics2=[]
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                freq2=np.array(freq2)
                freq2=np.reciprocal(freq2/fs)
                total = np.column_stack((freq2,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m) + ".txt", total, header="First column is the pitch value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                #Here it will give you the possibility of computing the correlations and Bootstrapping
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                total1=np.column_stack((freq2,spikespremot))
                total2=np.column_stack((freq2,spikesdur))
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                #This will get the data for Pitch vs Premotor
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Pitch column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Pitch syllable "+Syls[obj] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Pitch_Pre_syb"+ Syls[obj] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")						
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Pitch syllable "+Syls[obj] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Pitch_Pre_syb"+ Syls[obj] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    fichier.writelines("Syllable " + Syls[obj] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    os.chdir("..")
                    print(final)
                #This will get the data for Pitch vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Pitch column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Pitch syllable "+Syls[obj] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Pitch_During_syb"+ Syls[obj] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Pitch syllable "+Syls[obj] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Pitch_During_syb"+ Syls[obj] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    fichier.writelines("Syllable " + Syls[obj] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    os.chdir("..")
                    print(final)                  
                a2.set_xlabel("Number of Lags")
                a2.set_ylabel("Autocorrelation score")
                a3.hist(freq2, bins=int(np.mean(freq2)*0.01))
                a3.set_xlabel("Frequency (Hz)")
                a1.set_xlabel("Number of Lags")
                a1.set_ylabel("Autocorrelation score")
                a1.set_label(tellme("Now let's select the frequency. Key click (x2) for yes, mouse click for no")) #Here you will be asked to select a point in the peak that could represent the frequency (just to get an estimation)
                if not py.waitforbuttonpress(30):
                    os.chdir("Figures")
                    py.savefig("Corr_Pitch_syb"+ Syls[obj] +"_tone"+ str(m)+".jpg")
                    py.close()
                    os.chdir("..")
                    continue            
                else:
                    py.waitforbuttonpress(30)
                    while True:
                        freq = []
                        while len(freq) < 1:
                            tellme("Select the point for the frequency.")
                            freq = np.asarray(py.ginput(1, timeout=-1, show_clicks=True))
                        scat= a1.scatter(freq[:,0],freq[:,1], s=50, marker="X", zorder=10, c="b") 
                        ann=a1.annotate(str(int(np.reciprocal(freq[:,0]/fs))) +" Hz", xy=(freq[:,0],freq[:,1]), xytext=(freq[:,0]*1.2,freq[:,1]*1.2),
                                    arrowprops=dict(facecolor="black", shrink=0.05))
                                    
                        tellme("Happy? Key click for yes, mouse click for no")
                        if py.waitforbuttonpress(30):
                            os.chdir("Figures")
                            py.savefig("Corr_Pitch_syb"+ Syls[obj] +"_tone"+ str(m)+".jpg")
                            py.close()
                            os.chdir("..")
                            break
                        else:
                            ann.remove()
                            scat.remove()

			        
## 
#
# This function can be used to obtain the amplitude and its correlations of specific tones inside a syllable.
# It will allow you to work with the means or the area under the curve (integration)
# Deprecated: no correlation plot outputs, asks how to cut the syllable....
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate.
#
# means is the .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corramplitude(songfile, motifile, spikefile, fs=fs, window_size=window_size, means=None):
    
    #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile,0)  
       
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=finallist[0]
    elif answer.lower() == "b":
        used=finallist[1]
    elif answer.lower() == "c":
        used=finallist[2]    
    elif answer.lower() == "d":
        used=finallist[3]
    
    if means is not None:
        means = np.loadtxt(means).astype(int)
        syb=song[int(used[0][0]):int(used[0][1])]
        pass
    else: 
        #Will plot an exmaple of the syllable for you to get an idea of the number of chunks
        fig, az = py.subplots()
        example=song[int(used[0][0]):int(used[0][1])]
        tempo=np.linspace(used[0][0]/fs, used[0][1]/fs, len(example))
        abso=abs(example)
        az.plot(tempo,example)
        az.plot(tempo,abso)
        smooth=smoothed(np.ravel(example),fs)
        az.plot(tempo[:len(smooth)],smooth)
        az.set_title("Click on graph to move on.")
        py.waitforbuttonpress(10)
        numcuts=int(input("Number of chunks?"))
        py.close()
        
        # Will provide you 4 random exmaples of syllables to stablish the cutting points
        coords2=[]
        for j in range(4):           
           j=random.randint(0,len(used)-1)
           fig, ax = py.subplots()
           syb=song[int(used[j][0]):int(used[j][1])]
		   
		   #original by Eduarda
           #abso=abs(syb)
           #ax.plot(abso)
           #rms=window_rms(np.ravel(syb),window_size)
           #ax.plot(rms)
		   
           syb=syb[:,0]
           syb_smth=bandpass_filtfilt(syb, fs) #high pass( 500Hz) filter syllable
           abso_smth=abs(syb_smth)
           ax.plot(abso_smth)
           rms_smth=window_rms(np.ravel(syb_smth),window_size)
           ax.plot(rms_smth)

           py.waitforbuttonpress(10)
           while True:
               coords = []
               while len(coords) < numcuts+1:
                   tellme("Select the points to cut with mouse")
                   coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
               scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
               tellme("Happy? Key click for yes, mouse click for no")
               if py.waitforbuttonpress():
                   break
               else:
                   scat.remove()
           py.close()
           coords2=np.append(coords2,coords[:,0])
        
        #Will keep the mean coordinates for the cuts
        coords2.sort()
        coords2=np.split(coords2,numcuts+1)
        means=[]
        for k in range(len(coords2)):
            means+=[int(np.mean(coords2[k]))]
        np.savetxt("Mean_cut_syb"+answer+".txt", means) 
    
    # Will plot how the syllables will be cut according to the avarage of the coordinates clicked before by the user
    py.plot(syb_smth)
    for l in range(1,len(means)):
        py.plot(np.arange(means[l-1],means[l-1]+len(syb_smth[means[l-1]:means[l]])),syb_smth[means[l-1]:means[l]])   

    # Autocorrelation and Distribution 
    an2=input("Want to execute correlations with Means or Integration?")
    for m in range(1,len(means)):
        spikespremot=[]
        spikesdur=[]
        amps=[]
        integ=[]
        fig=py.figure(figsize=(25,12))
        gs=py.GridSpec(2,3)
        a1=fig.add_subplot(gs[0,:]) # First row, first column
        a2=fig.add_subplot(gs[1,0]) # First row, second column
        a3=fig.add_subplot(gs[1,1])
        a4=fig.add_subplot(gs[1,2])
        statistics=[]
        statistics2=[]
        for n in range(len(used)):
            syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
            sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
            smooth=smoothed(np.ravel(sybcut),fs)
            beg=(used[n][0] + means[m-1])/fs
            end=(used[n][0] + means[m])/fs
            step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
            step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
            spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
            spikesdur+=[[np.size(step2)/(end-beg)]]
            amps+=[np.mean(smooth)]
            integ+=[scipy.integrate.simps(smooth)]
        a1.plot(abs(sybcut))
        a1.set_title("Syllable " + answer + " Tone " + str(m))
        a1.set_xlabel("Sample points")
        a1.set_ylabel("Amplitude")
        a1.fill_between(np.arange(0,len(smooth),1), 0, smooth, zorder=10, color="b", alpha=0.1)
        spikesdur=np.array(spikesdur)[:,0]
        spikespremot=np.array(spikespremot)[:,0]
        amps=np.array(amps)
        integ=np.array(integ)
        if an2[0].lower() == "m":
            total = np.column_stack((amps,spikespremot,spikesdur))
            np.savetxt("Data_Raw_Corr_Amplitude_Result_Syb" + answer + "_tone_" + str(m) + "_" + an2 + ".txt", total, header="First column is the amplitude value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
            total1=np.column_stack((amps,spikespremot))
            total2=np.column_stack((amps,spikesdur))
            a2.hist(amps)
            a2.set_title("Distribution of the Raw Means")
            a2.set_ylabel("Frequency")
            a2.set_xlabel("Mean Values")
        else:
            total = np.column_stack((integ,spikespremot,spikesdur))
            np.savetxt("Data_Raw_Corr_Amplitude_Result_Syb" + answer + "_tone_" + str(m)+ "_" + an2 + ".txt", total, header="First column is the amplitude value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
            total1=np.column_stack((integ,spikespremot))
            total2=np.column_stack((integ,spikesdur))
            a2.hist(integ)
            a2.set_title("Distribution of the Raw Integration")
        #Here it will give you the possibility of computing the correlations and Bootstrapping
        an=input("Correlations?")
        if an.lower() == "n":
            pass
        else:
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            z1 = np.abs(scipy.stats.zscore(total1))
            z2 = np.abs(scipy.stats.zscore(total2))
            total1=total1[(z1 < threshold).all(axis=1)]
            total2=total2[(z2 < threshold).all(axis=1)]
            a = total1[:,1] == 0
            b = total2[:,1] == 0
            if len(total1) < 3 or all(a) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total1[:,0])[1] #Amplitude column
                s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                homo=scipy.stats.levene(total1[:,0],total1[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                #This will get the data for Amplitude vs Premotor
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + answer + "_tone_" + str(m)+ "_Premotor_"+ an2 +".txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                print(final)
                a3.hist(np.array(statistics)[:,0])
                a3.set_title("Bootstrap Premotor")
                a3.set_xlabel("Correlation Values")
            #This will get the data for Pitch vs During     
            if len(total2) < 3 or all(b) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total2[:,0])[1] #Amplitude column
                s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + answer + "_tone_" + str(m)+ "_During_" + an2 + ".txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                a4.hist(np.array(statistics2)[:,0])
                a4.set_title("Bootstrap During")
                a4.set_xlabel("Correlation Values")
                print(final)
                py.savefig(fname="Corr_Amplitude_syb"+ answer +"_tone"+ str(m) +".tif")

	
## 
#
# This function can be used to obtain the amplitude and its correlations of specific tones inside a syllable.
# It will allow you to work with the means or the area under the curve (integration). Based on corramplitude but without asking for syllable 
# chunks used for amplitude computation. Read edges of the chunks from files
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate.
#
# means is the .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corramplitude_auto(songfile, motifile, spikefile, fs=fs, window_size=window_size):
    
    #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile,0)  
    f = open("SummaryCorrAmp.txt", "w+")
    y=["MeanA.txt","MeanB.txt","MeanC.txt","MeanD.txt"]
    Syls=["a","b","c","d"]
    check=jumpsyl(spikefile)
    for g in range(len(finallist)):
        if Syls[g] in check:
            continue
        else:
            used=finallist[g]
            means = np.loadtxt(y[g]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
            
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                amps=[]
                fig=py.figure(figsize=(25,12))
                gs=py.GridSpec(2,3)
                a1=fig.add_subplot(gs[0,:]) # First row, first column
                a2=fig.add_subplot(gs[1,0]) # First row, second column
                a3=fig.add_subplot(gs[1,1])
                a4=fig.add_subplot(gs[1,2])
                statistics=[]
                statistics2=[]
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    smooth=smoothed(np.ravel(sybcut),fs)
                    beg=(used[n][0] + means[m-1])/fs
                    end=(used[n][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                    amps+=[np.mean(smooth)]
                a1.plot(abs(sybcut))
                a1.plot(smooth)
                a1.set_title("Syllable " + Syls[g] + " Tone " + str(m))
                a1.set_ylabel("Amplitude")
                a1.set_xlabel("Sample points")
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                amps=np.array(amps)
                total = np.column_stack((amps,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m) + "_Mean.txt", total, header="First column is the amplitude value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                total1=np.column_stack((amps,spikespremot))
                total2=np.column_stack((amps,spikesdur))
                a2.hist(amps)
                a2.set_title("Distribution of the Raw Means")
                a2.set_ylabel("Frequency")
                a2.set_xlabel("Mean Values")
                #Start of Correlations
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Amplitude column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    #This will get the data for Amplitude vs Premotor
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Amplitude syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Amplitude_Pre_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")

                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Amplitude syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Amplitude_Pre_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_Premotor_Mean.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    print(final)
                    a3.hist(np.array(statistics)[:,0])
                    a3.set_title("Bootstrap Premotor")
                    a3.set_xlabel("Correlation Values")
                #This will get the data for Amplitude vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Amplitude column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Amplitude syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Amplitude_During_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Amplitude syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Amplitude_During_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_During_Mean.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    a4.hist(np.array(statistics2)[:,0])
                    a4.set_title("Bootstrap During")
                    a4.set_xlabel("Correlation Values")
                    print(final)
                os.chdir("Figures")
                py.savefig(fname="Corr_Amplitude_syb"+ Syls[g] +"_tone"+ str(m) +".jpg")
                py.close()
                os.chdir("..")

	
##
# This function computes the Spectral Entropy of a signal. 
#The power spectrum is computed through fft. Then, it is normalised and assimilated to a probability density function.
#
# Arguments:
#    ----------
#    signal : list or array
#        List or array of values.
#    sampling_rate : int
#        Sampling rate (samples/second).
#    bands : list or array
#        A list of numbers delimiting the bins of the frequency bands. If None the entropy is computed over the whole range of the DFT (from 0 to `f_s/2`).
#
#    Returns
#    ----------
#    spectral_entropy : float
#        The spectral entropy as float value.
def complexity_entropy_spectral(signal, fs=fs, bands=None):
    """
    Based on the `pyrem <https://github.com/gilestrolab/pyrem>`_ repo by Quentin Geissmann.
    
    Example
    ----------
    >>> import neurokit as nk
    >>>
    >>> signal = np.sin(np.log(np.random.sample(666)))
    >>> spectral_entropy = nk.complexity_entropy_spectral(signal, 1000)

    Notes
    ----------
    *Details*

    - **Spectral Entropy**: Entropy for different frequency bands.


    *Authors*

    - Quentin Geissmann (https://github.com/qgeissmann)

    *Dependencies*

    - numpy

    *See Also*

    - pyrem package: https://github.com/gilestrolab/pyrem
    """

    psd = np.abs(np.fft.rfft(signal))**2
    psd /= np.sum(psd) # psd as a pdf (normalised to one)

    if bands is None:
        power_per_band= psd[psd>0]
    else:
        freqs = np.fft.rfftfreq(signal.size, 1/float(fs))
        bands = np.asarray(bands)

        freq_limits_low = np.concatenate([[0.0],bands])
        freq_limits_up = np.concatenate([bands, [np.Inf]])

        power_per_band = [np.sum(psd[np.bitwise_and(freqs >= low, freqs<up)])
                for low,up in zip(freq_limits_low, freq_limits_up)]

        power_per_band= np.array(power_per_band)[np.array(power_per_band) > 0]

    spectral = - np.sum(power_per_band * np.log2(power_per_band))
    return(spectral)

## 
#
# This function can be used to obtain the spectral entropy and its correlations of specific tones inside a syllable.
# Deprecated: no correlation plot outputs, asks how to cut the syllable....
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate (Hz)
#
# means is the a .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corrspectral(songfile, motifile, spikefile, fs=fs,  window_size=window_size, means=None):
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    
    finallist=sortsyls(motifile,0)  
    
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=finallist[0]
    elif answer.lower() == "b":
        used=finallist[1]
    elif answer.lower() == "c":
        used=finallist[2]    
    elif answer.lower() == "d":
        used=finallist[3]
    
    if means is not None:
        means = np.loadtxt(means).astype(int)
        syb=song[int(used[0][0]):int(used[0][1])]
        pass
    else: 
        #Will plot an exmaple of the syllable for you to get an idea of the number of chunks
        fig, az = py.subplots()
        example=song[int(used[0][0]):int(used[0][1])]
        tempo=np.linspace(used[0][0]/fs, used[0][1]/fs, len(example))
        abso=abs(example)
        az.plot(tempo,example)
        az.plot(tempo,abso)
        smooth=smoothed(np.ravel(example), fs)
        az.plot(tempo[:len(smooth)],smooth)
        az.set_title("Click on graph to move on.")
        py.waitforbuttonpress(10)
        numcuts=int(input("Number of chunks?"))
        py.close()
        
        # Will provide you 4 random exmaples of syllables to stablish the cutting points
        coords2=[]
        for j in range(4):           
           j=random.randint(0,len(used)-1)
           fig, ax = py.subplots()
           syb=song[int(used[j][0]):int(used[j][1])]
		   
		   #original by Eduarda
           #abso=abs(syb)
           #ax.plot(abso)
           #rms=window_rms(np.ravel(syb),window_size)
           #ax.plot(rms)
		   
           syb=syb[:,0]
           syb_smth=bandpass_filtfilt(syb, fs) #high pass( 500Hz) filter syllable
           abso_smth=abs(syb_smth)
           ax.plot(abso_smth)
           rms_smth=window_rms(np.ravel(syb_smth),window_size)
           ax.plot(rms_smth)
		   
           py.waitforbuttonpress(10)
           while True:
               coords = []
               while len(coords) < numcuts+1:
                   tellme("Select the points to cut with mouse")
                   coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
               scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
               tellme("Happy? Key click for yes, mouse click for no")
               if py.waitforbuttonpress():
                   break
               else:
                   scat.remove()
           py.close()
           coords2=np.append(coords2,coords[:,0])
        
        #Will keep the mean coordinates for the cuts
        coords2.sort()
        coords2=np.split(coords2,numcuts+1)
        means=[]
        for k in range(len(coords2)):
            means+=[int(np.mean(coords2[k]))]
        np.savetxt("Mean_cut_syb"+answer+".txt", means) 
    
    # Will plot how the syllables will be cut according to the avarage of the coordinates clicked before by the user
    py.plot(syb_smth)
    for l in range(1,len(means)):
        py.plot(np.arange(means[l-1],means[l-1]+len(syb_smth[means[l-1]:means[l]])),syb_smth[means[l-1]:means[l]])   

    # Autocorrelation and Distribution 
    for m in range(1,len(means)):
        spikespremot=[]
        spikesdur=[]
        specent=[]
        fig=py.figure(figsize=(25,12))
        gs=py.GridSpec(1,3)
        a2=fig.add_subplot(gs[0,0]) # First row, second column
        a3=fig.add_subplot(gs[0,1])
        a4=fig.add_subplot(gs[0,2])
        statistics=[]
        statistics2=[]
        for n in range(len(used)):
            syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
            sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
            SE=complexity_entropy_spectral(sybcut[:,0],fs)
            beg=(used[n][0] + means[m-1])/fs
            end=(used[n][0] + means[m])/fs
            step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
            step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
            spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
            spikesdur+=[[np.size(step2)/(end-beg)]]
            specent+=[[SE]]
        fig.suptitle("Syllable " + answer + " Tone " + str(m))
        spikesdur=np.array(spikesdur)[:,0]
        spikespremot=np.array(spikespremot)[:,0]
        specent=np.array(specent)
        total = np.column_stack((specent,spikespremot,spikesdur))
        np.savetxt("Data_Raw_Corr_SpecEnt_Result_Syb" + answer + "_tone_" + str(m) + ".txt", total, header="First column is the spectral value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
        #Here it will give you the possibility of computing the correlations and Bootstrapping
        an=input("Correlations?")
        if an.lower() == "n":
            pass
        else:
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            total1=np.column_stack((specent,spikespremot))
            total2=np.column_stack((specent,spikesdur))
            z1 = np.abs(scipy.stats.zscore(total1))
            z2 = np.abs(scipy.stats.zscore(total2))
            total1=total1[(z1 < threshold).all(axis=1)]
            total2=total2[(z2 < threshold).all(axis=1)]
            a = total1[:,1] == 0
            b = total2[:,1] == 0
            a2.hist(specent)
            a2.set_title("Distribution of the Raw Spectral Entropy")
            a2.set_ylabel("Frequency")
            a2.set_xlabel("Spectral Values")
            #This will get the data for Spectral Entropy vs Premotor
            if len(total1) < 3 or all(a) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total1[:,0])[1] #Spectral Entropy column
                s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                homo=scipy.stats.levene(total1[:,0],total1[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total1[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_SpecEnt_Result_Syb" + answer + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                print(final)
                a3.hist(np.array(statistics)[:,0])
                a3.set_title("Bootstrap Premotor")
                a3.set_xlabel("Correlation Values")
            #This will get the data for Spectral Entropy vs During     
            if len(total2) < 3 or all(b) == True:
                pass
            else:
                s1=scipy.stats.shapiro(total2[:,0])[1] #Spectral Entropy column
                s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                comb1=np.array([s1,s2,homo])
                comb1=comb1>alpha
                if  comb1.all() == True: #test for normality
                    final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics2+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                        res=scipy.stats.spearmanr(total2[:,1],resample)
                        statistics2+=[[res[0],res[1]]]
                np.savetxt("Data_Boot_Corr_SpectEnt_Result_Syb" + answer + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")    
                print(final)
                a4.hist(np.array(statistics2)[:,0])
                a4.set_title("Bootstrap During")
                a4.set_xlabel("Correlation Values")
                py.savefig("Corr_SpecEnt_syb"+ answer +"_tone"+ str(m)+".tif")


## 
#
# This function can be used to obtain the spectral entropy and its correlations of specific tones inside a syllable. Based on corrspectral but without asking for syllable 
# chunks used for spectral computation. Read edges of the chunks from files
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate
#
# means is the a .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corrspectral_auto(songfile, motifile, spikefile, fs=fs, window_size=window_size):
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile,0)  
    f = open("SummaryCorrSpecEnt.txt", "w+")
    y=["MeanA.txt","MeanB.txt","MeanC.txt","MeanD.txt"]
    Syls=["a","b","c","d"]
    check=jumpsyl(spikefile)  
    for g in range(len(finallist)):
        if Syls[g] in check:
            continue
        else:
            used=finallist[g]
            means = np.loadtxt(y[g]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                specent=[]
                fig=py.figure(figsize=(25,12))
                gs=py.GridSpec(1,3)
                a2=fig.add_subplot(gs[0,0]) # First row, second column
                a3=fig.add_subplot(gs[0,1])
                a4=fig.add_subplot(gs[0,2])
                statistics=[]
                statistics2=[]
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    SE=complexity_entropy_spectral(sybcut[:,0],fs)
                    beg=(used[n][0] + means[m-1])/fs
                    end=(used[n][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                    specent+=[[SE]]
                fig.suptitle("Syllable " + Syls[g] + " Tone " + str(m))
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                specent=np.array(specent)
                total = np.column_stack((specent,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m) + ".txt", total, header="First column is the spectral value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                #Here it will give you the possibility of computing the correlations and Bootstrapping
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                total1=np.column_stack((specent,spikespremot))
                total2=np.column_stack((specent,spikesdur))
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                a2.hist(specent)
                a2.set_title("Distribution of the Raw Spectral Entropy")
                a2.set_ylabel("Frequency")
                a2.set_xlabel("Spectral Values")
                #This will get the data for Spectral Entropy vs Premotor
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Spectral Entropy column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Entropy syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Entropy_Pre_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
						
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total1[:,0]).reshape(-1,1)
                           y_fr=(total1[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Entropy syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Entropy_Pre_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
						
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    print(final)
                    a3.hist(np.array(statistics)[:,0])
                    a3.set_title("Bootstrap Premotor")
                    a3.set_xlabel("Correlation Values")
                #This will get the data for Spectral Entropy vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Spectral Entropy column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Entropy syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Entropy_During_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        if(final[1]<alpha):
                           # Create linear regression object
                           regr = LinearRegression()
                           # Train the model using the training sets
                           x_dur=(total2[:,0]).reshape(-1,1)
                           y_fr=(total2[:,1]).reshape(-1,1)
                           fr_mean=np.mean(y_fr)
                           y_fr=y_fr-fr_mean
                           regr.fit(x_dur, y_fr)
                           # Make predictions using the testing set
                           fr_pred = regr.predict(x_dur)
                           py.fig, ax = py.subplots(1, figsize=(25,12))
                           ax.plot(x_dur,y_fr,'bo')
                           ax.plot(x_dur,fr_pred,'r')
                           ax.set_xlabel("Entropy syllable "+Syls[g] +"_tone"+ str(m)+" (seconds)",fontsize=18)
                           ax.set_ylabel("Firing rate deviation (Hz)",fontsize=18)	
                           ax.set_title("Correlation: "+str(final[0])+",  p-value: "+str(final[1]),fontsize=18)
                           ax.tick_params(
                               axis="both",          # changes apply to the x-axis
                               which="major",      # both major and minor ticks are affected
                               labelsize=16)	
                           wind=py.get_current_fig_manager()
                           wind.window.showMaximized()
					       #save above figure
                           os.chdir("Figures")
                           py.savefig("Corr_Entropy_During_syb"+ Syls[g] +"_tone"+ str(m)+"_scatter.jpg")
                           py.close()
                           os.chdir("..")	
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")   
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    print(final)
                    a4.hist(np.array(statistics2)[:,0])
                    a4.set_title("Bootstrap During")
                    a4.set_xlabel("Correlation Values")
                os.chdir("Figures")
                py.savefig("Corr_SpecEnt_syb"+ Syls[g] +"_tone"+ str(m)+".jpg")
                py.close()
                os.chdir("..")
    
				
				
				
## 
#
# This function can be used to obtain the ISI from the whole recording
#
# Arguments:
#
# spikefile is the .txt file containing the times of the spikes
#
def ISI(spikefile):
    spikes=np.loadtxt(spikefile)
    times=np.sort(np.diff(spikes))*1000
    py.figure()
    py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    py.xscale('log')
    py.xlabel("Millisecond (ms)")
    py.ylabel("Counts/bin")
    py.title("ISI global")
	

## 
#
# This function can be used to obtain the ISI only from the relevant portions of the recording: the portion used for baseline computations (silence period) and
# the portions during the motif renditions
#
# Arguments:
#
# spikefile is the .txt file containing the times of the spikes
#
# basebeg is the begin of the window for baseline computation of the fr
#
# basend is the end of the window for baseline computation of the fr
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
def ISI_relevant(spikefile,motifile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    if(len(noisy_motifs)!=0):
       all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)
    else:
       all_motifs=clean_motifs

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	
	###################################
	# Plot ISI for motif period
	###################################
	
    shoulder_beg= 0.02 #in seconds
    shoulder_end= 0.02 #in seconds
    spikes_motif=[]

    nb_motifs=len(all_motifs[:,0]) #number of sung motifs (noisy or not)
    for i in range(nb_motifs):
        motif_on=(all_motifs[i,0]/fs)-shoulder_beg#onst of motif
        motif_off=(all_motifs[i,-1]/fs)+shoulder_end#offset of motif
        spikes_motif+=[spused[np.where(np.logical_and(spused >= motif_on, spused <= motif_off) == True)]]
    
    spikes_motif=np.concatenate(spikes_motif[:])
    times=np.sort(np.diff(spikes_motif))*1000
    times=times[np.where(np.logical_and(times >= 0, times <= 1000) == True)]

    py.fig, ax = py.subplots(1, figsize=(25,12))
    ax.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    ax.set_xscale('log')
    ax.set_xlabel("Time(ms)",fontsize=30)
    ax.set_ylabel("Counts/bin",fontsize=30)	
    ax.set_title("ISI motif",fontsize=30)
    ax.tick_params(
        axis="both",          # changes apply to the x-axis
        which="major",      # both major and minor ticks are affected
        labelsize=30)	
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()	
	
    #py.figure()
    #py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    #py.xscale('log')
    #py.xlabel("Millisecond (ms)",fontsize=18)
    #py.ylabel("Counts/bin",fontsize=18)	
    #py.title("ISI motif",fontsize=18)
    #py.tick_params(
    #    axis="both",          # changes apply to the x-axis
    #    which="major",      # both major and minor ticks are affected
    #    labelsize=16)
	
	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times >= 0, times <= 1000) == True)]
	
    py.fig, ax = py.subplots(1, figsize=(25,12))
    #ax.set_xlim(0, 10e3)
    ax.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    ax.set_xscale('log')
    ax.set_xlabel("Time(ms)",fontsize=30)
    ax.set_ylabel("Counts/bin",fontsize=30)	
    ax.set_title("ISI baseline",fontsize=30)
    ax.tick_params(
        axis="both",          # changes apply to the x-axis
        which="major",      # both major and minor ticks are affected
        labelsize=30)	
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    #py.xscale('log')
    #py.xlabel("Millisecond (ms)",fontsize=18)
    #py.ylabel("Counts/bin",fontsize=18)	
    #py.title("ISI baseline",fontsize=18)	
    #py.tick_params(
    #    axis="both",          # changes apply to the x-axis
    #    which="major",      # both major and minor ticks are affected
    #    labelsize=16)

## 
#
# As ISI_relevant but plots only for baseline period
#
#	
def ISI_baseline(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times >= 0, times <= 1000) == True)]
	
    py.fig, ax = py.subplots(1, figsize=(25,12))
    #ax.set_xlim(0, 10e3)
    if(len(times)<1):
      times=[1,10]
    ax.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    ax.set_xscale('log')
    ax.set_xlabel("Time(ms)",fontsize=30)
    ax.set_ylabel("Counts/bin",fontsize=30)	
    ax.set_title("ISI baseline",fontsize=30)
    ax.tick_params(
        axis="both",          # changes apply to the x-axis
        which="major",      # both major and minor ticks are affected
        labelsize=30)	
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    #py.xscale('log')
    #py.xlabel("Millisecond (ms)",fontsize=18)
    #py.ylabel("Counts/bin",fontsize=18)	
    #py.title("ISI baseline",fontsize=18)	
    #py.tick_params(
    #    axis="both",          # changes apply to the x-axis
    #    which="major",      # both major and minor ticks are affected
    #    labelsize=16)

## 
#
# As ISI_relevant but plots only for baseline period
#
#		
def ISI_motif(spikefile,motifile,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    finallist=sortsyls_psth_glob(motifile,0)
    clean_motifs=np.array(finallist[0])
    noisy_motifs=np.array(finallist[1])
    if(len(noisy_motifs)!=0):
       all_motifs=np.concatenate((np.array(finallist[0]),np.array(finallist[1])),axis=0)
    else:
       all_motifs=clean_motifs

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	
	###################################
	# Plot ISI for motif period
	###################################
	
    shoulder_beg= 0.02 #in seconds
    shoulder_end= 0.02 #in seconds
    spikes_motif=[]

    nb_motifs=len(all_motifs[:,0]) #number of sung motifs (noisy or not)
    for i in range(nb_motifs):
        motif_on=(all_motifs[i,0]/fs)-shoulder_beg#onst of motif
        motif_off=(all_motifs[i,-1]/fs)+shoulder_end#offset of motif
        spikes_motif+=[spused[np.where(np.logical_and(spused >= motif_on, spused <= motif_off) == True)]]
    
    spikes_motif=np.concatenate(spikes_motif[:])
    times=np.sort(np.diff(spikes_motif))*1000
    times=times[np.where(np.logical_and(times >= 0, times <= 1000) == True)]

    py.fig, ax = py.subplots(1, figsize=(25,12))
    if(len(times)<1):
      times=[1,10]
    #print(times)
    ax.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    ax.set_xscale('log')
    ax.set_xlabel("Time(ms)",fontsize=30)
    ax.set_ylabel("Counts/bin",fontsize=30)	
    ax.set_title("ISI motif",fontsize=30)
    ax.tick_params(
        axis="both",          # changes apply to the x-axis
        which="major",      # both major and minor ticks are affected
        labelsize=30)	
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()	
	
    #py.figure()
    #py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    #py.xscale('log')
    #py.xlabel("Millisecond (ms)",fontsize=18)
    #py.ylabel("Counts/bin",fontsize=18)	
    #py.title("ISI motif",fontsize=18)
    #py.tick_params(
    #    axis="both",          # changes apply to the x-axis
    #    which="major",      # both major and minor ticks are affected
    #    labelsize=16)
	
	

	
	
## 
#
# As ISI_relevant but plots only for baseline period
#
#	
def log_ISI_baseline(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times > 0, times <= 1000) == True)]
    log_times=np.log(times)
	
    py.fig, ax = py.subplots(1, figsize=(25,12))
    #ax.set_xlim(0, 10e3)
    if(len(times)<1):
      times=[1,10]
    ax.hist(log_times, bins= 200)
    #ax.set_xscale('log')
    ax.set_xlabel("Time(ms)",fontsize=30)
    ax.set_ylabel("Counts/bin",fontsize=30)	
    ax.set_title("log ISI baseline",fontsize=30)
    ax.tick_params(
        axis="both",          # changes apply to the x-axis
        which="major",      # both major and minor ticks are affected
        labelsize=30)	
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    #py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    #py.xscale('log')
    #py.xlabel("Millisecond (ms)",fontsize=18)
    #py.ylabel("Counts/bin",fontsize=18)	
    #py.title("ISI baseline",fontsize=18)	
    #py.tick_params(
    #    axis="both",          # changes apply to the x-axis
    #    which="major",      # both major and minor ticks are affected
    #    labelsize=16)	

## 
#
# Computes the CV of the firing during baseline period
#
#	
def cv_baseline_fr(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times >= 0, times <= 1000) == True)]
	
    mean_=np.mean(times)
    std_=np.std(times)
    cv=std_/mean_
    return cv


## 
#
# Computes the LCV of the firing during baseline period (the cv of the distribution of the log of ISI)
#
#	
def cv_log_baseline_fr(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times > 0, times <= 1000) == True)]
    log_times=np.log(times)
	
    mean_=np.mean(log_times)
    std_=np.std(log_times)
    cv=std_/mean_
    return cv
	
## 
#
# Computes the LCV of the firing during baseline period (the cv of the distribution of the log of ISI)
#
#	
def entropy_log_baseline_fr(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times > 0, times <= 1000) == True)]
    log_times=np.log(times)
    hist,bin_edg = np.histogram(log_times, bins= 1000)
    #print(len(np.arange(np.min(log_times), np.max(log_times), 1)))
    entrpy_log=scipy.stats.entropy(hist)
	
    return entrpy_log		
	
## 
#
# Computes the skewness of the firing during baseline period 
#
#	
def skewness_ISI_baseline_fr(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times > 0, times <= 1000) == True)]
    hist,bin_edg = np.histogram(times, bins= 1000)
    #print(len(np.arange(np.min(log_times), np.max(log_times), 1)))
    skwness=scipy.stats.skew(hist)
	
    return skwness			

## 
#
# Computes the kurtosis of the firing during baseline period 
#
#	
def kurtosis_ISI_baseline_fr(spikefile,basebeg,basend,fs=fs):
    #sybs=["a","b","c","d"]
    #index of the noisy syllable (the syllable that received the noise on top of itself), by convention it comes after all relevant 
	#syllables (e.g. if motif is a,b,c,d and the syll c receives noise, the labels will be a,b,c,d,e with e being noisy c)
	#idx to be set by the user. It is never 0. index of the clean syllable (the one that receives probabilistic noise). Later try to ask for both indeces in the console

    #idx_noisy_syb = 2 #idex in syb of the relevant syb that probabilistically receives noise and that is labelled using the last label in syb
    #len_motif=len(sybs)-1 #length of the motif (nb syllables)

    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
	

	###################################
	# Plot ISI for baseline period
	###################################

    spikes_baseline=[]
    spikes_baseline=[spused[np.where(np.logical_and(spused >= basebeg, spused <= basend) == True)]]
    spikes_baseline=np.array(spikes_baseline)

    times=np.sort(np.diff(spikes_baseline))*1000
    times=times[np.where(np.logical_and(times > 0, times <= 1000) == True)]
    hist,bin_edg = np.histogram(times, bins= 1000)
    #print(len(np.arange(np.min(log_times), np.max(log_times), 1)))
    kurtosis_=scipy.stats.kurtosis(hist)
	
    return kurtosis_			
		
