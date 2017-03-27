# -*- coding: utf-8 -*-
"""
Created on Fri Mai 29 11:26:54 2015

@author: GUAN Wei
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
import matplotlib.pyplot as plt
from scipy.io import wavfile
#import sounddevice as sd
import scipy as sc
import math

plt.close('all')
########################################################################""
def filtre(z0,p,k,taille):
    la=np.arange(0,1.0,1.0/taille) #les frequences reduites
    #z= #definition de la variable z
    taillez=z0.shape[0]  #le nbr de zeros
    taillep=p.shape[0]   #le nbr de poles
    #......
    return H
#########################################################################"""



#Calcul des fréquences en fonction du La de référence
# pour comparer avec les fréquences présentes dans le signal
#pour changer d'octave il suffit de faire La*(2^n)
#vous pouvez mettre en commentaire ensuite

La=440
k=np.arange(-12,13,1)
r=np.power(np.float64(2),np.float64(1.0/12))
f=La*np.power(r,(k))
#####################################################################""



sampFreq, sndinit = wavfile.read('Une_note.wav')
snd = np.float64(sndinit) / (2.**15) #normalisation des données, fichier son 16 bits entiers, il faut transformer en float pour les calculs
[Nbr,channel]=snd.shape #récupère le nbr d'échantillon, et le nombre de canaux

#Nbr=snd.size #quand c'est un signal mono

#ws.PlaySound(np.string0(sndinit),ws.SND_MEMORY)
#Nbr= 256*4 #valeur à étudier

Nbr=np.floor(16/(440.0/sampFreq))

#snd.play(sndinit, sampFreq)  #faire jouer le tableau numpy de son donc le signal brut
#snd.stop()
#snd.wait()
#np.save("son_son",snd)
#snd=np.load("son_son.npy")

#sd.wait()
np.save("son_numpy",snd)
snd=np.load("son_numpy.npy")
#s1 = snd #dans le cas mono
s1 = snd[2500:2500+Nbr,0] #on ne prend que le premier canal pour signal
timeArray = np.arange(0, Nbr, 1) #on fait le vecteur temps
timeArray = np.float64(timeArray) / sampFreq
titimeArray = timeArray * 1000  #scale to milliseconds
plt.figure(1)
plt.plot(timeArray, s1, color='b')
plt.ylabel('Amplitude')
plt.xlabel('Time (ms)')
plt.grid()
plt.draw()

"signal"
#x = np.linspace(0,N-1,N) #autre méthode

plt.figure()
plt.plot(s1),plt.title(['s1 ', np.str(Nbr), ' points']);

#fft du signal"

#tracez la réponse fréquentielle

n = len(s1) 
p = sc.fft(s1) # take the fourier transform 
n2=len(p)
#nUniquePts = np.ceil((n+1)/2.0)
#p = p[0:nUniquePts]
p = abs(p)
#plt.stem()

plt.figure(2)
plt.plot(p),plt.title(['p ', np.str(Nbr), ' en frequence']);


plt.figure(3)
Array = np.arange(0, Nbr, 1) #on fait le vecteur indice
pasf=1.0/Nbr
plt.plot(Array*pasf*sampFreq, p, color='b')
plt.ylabel('Amplitude')
plt.xlabel('Frequence (Hz)')
plt.grid()
plt.draw()



plt.figure(4)
plt.plot(Array*pasf*sampFreq, 20*np.log10(np.abs(p)), color='b')
plt.ylabel('Amplitude')
plt.xlabel('Frequence (Hz)')
plt.grid()
plt.draw()

#Etude de la fft après multiplication du signal temporel par des fenêtres d'apodisation
#fen2 = np.blackman(N);
#fen3 = np.ones(N);

##fenetre de hamming"
N=Nbr
fen1 = np.hamming(N);
fen = np.reshape(fen1,N)
sin1fen = s1*fen;
#sin2fen = sin2*fen;
#
#Tracez les réponses fréquentielles pour chaque fenêtre, bilan?
