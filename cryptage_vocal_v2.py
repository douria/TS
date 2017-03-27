# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:04:10 2015

@author: ladretp
"""
#Definition of local functions
#######To clear the working memory###########
def clearall():
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]
#############################################
        
clearall()   

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy.io import wavfile
import sounddevice as sd



plt.close('all')
plt.isinteractive()



sampFreq, sndinit = wavfile.read('grenoble.wav') #le fichier a été enregistré avec une fe=22050 Hz
snd = np.float64(sndinit) / (2.**15) #normalisation des données, fichier son 16 bits entiers, il faut transformer en float pour les calculs
#[Nbr,channel]=snd.shape #récupère le nbr d'échantillon, et le nombre de canaux
Nbr=snd.size #quand c'est un signal mono
#s1 = snd[:,0] #on ne prend que le premier canal pour signal
#ws.PlaySound(np.string0(sndinit),ws.SND_MEMORY)

sd.play(sndinit, sampFreq)  #faire jouer le tableau numpy de son donc le signal brut
#sd.stop()
sd.wait()
np.save("son_numpy",snd)
snd=np.load("son_numpy.npy")
s1 = snd
timeArray = np.arange(0, Nbr, 1) #on fait le vecteur temps
timeArray = np.float64(timeArray) / sampFreq
titimeArray = timeArray * 1000  #scale to milliseconds
plt.figure(1)
plt.plot(timeArray, s1, color='b')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')
plt.grid()
plt.draw()

n = len(s1) 
p = sc.fft(s1) # take the fourier transform 
n2=len(p)
nUniquePts = np.ceil((n+1)/2.0)
p = p[0:nUniquePts]
p = abs(p)

#p = p / float(n) # scale by the number of points so th
                 # the magnitude does not depend on the length 
                 # of the signal or on its sampling frequency  
p = p**2  # square it to get the power 

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
if n % 2 > 0: # we've got odd number of points fft
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

freqArray = np.arange(0, nUniquePts, 1.0) * (np.float64(sampFreq )/ n); # (nbr pt )*f/fe
#freqArray = arange(0, nUniquePts, 1.0)
plt.figure(2)
plt.plot(freqArray/1000, 10*np.log10(p), color='g')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.grid()
plt.draw()
############################################################################
# cryptage
#######################################################################
#g(k)=(1)^^k*x(k)
unk=(-1)*np.ones(n)
ind=np.arange(0,n)
unk=unk**ind
g=unk*s1
plt.figure(3)
plt.plot(timeArray, g, color='b')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')
plt.title('signal crypte')
plt.grid()
plt.draw()
g2=np.int16(g*2**15)
sd.play(g, sampFreq)
#sd.stop()
sd.wait()

    
 #  le spectre du signal crypté
    
n = len(s1) 
p = sc.fft(g) # take the fourier transform 
n2=len(p)
nUniquePts = np.ceil((n+1)/2.0)
p = p[0:nUniquePts]
p = abs(p)

#p = p / float(n) # scale by the number of points so that
                 # the magnitude does not depend on the length 
                 # of the signal or on its sampling frequency  
p = p**2  # square it to get the power 

# multiply by two (see technical document for details)
# odd nfft excludes Nyquist point
if n % 2 > 0: # we've got odd number of points fft
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

freqArray = np.arange(0, nUniquePts, 1.0) * (np.float64(sampFreq )/ n); # (nbr pt )*f/fe
#freqArray = arange(0, nUniquePts, 1.0)
plt.figure(4)
plt.plot(freqArray/1000, 10*np.log10(p), color='g')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.title('spectre crypte')
plt.grid()
#plt.isinteractive()

#plt.draw()

#raw_input('tapez une touche pour continuer')


############################################################################
# décryptage
#######################################################################
#g(k)=(1)^^k*x(k)
unk=(-1)*np.ones(n)
ind=np.arange(0,n)
unk=unk**ind
g=unk*g #on reprend le g précédent et on remultiplie par -1


sd.play(g, sampFreq)
sd.wait()
