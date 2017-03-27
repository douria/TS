# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:30:32 2017

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
import skimage as sk
from skimage import color, data, feature
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, fftshift,ifft2
from scipy.ndimage import convolve
from scipy.signal import filtfilt, lfilter,lfilter_zi
from matplotlib import colors as mp

###############################################################"
            

plt.close('all')


###
N0=256
Tx=4
Ty=16

x=np.arange(0,N0,1)
y=np.arange(0,N0,1)
X,Y=np.meshgrid(x,y)

I=50*np.sin(2*np.pi*Tx/N0*X+2*np.pi*Ty/N0*Y); 

plt.figure(2)
plt.imshow(I,cmap='gray')


###########################################
#Calcul de la fft et affichage centré
###########################################
fftI=fft2(I)

plt.figure(3)
plt.imshow(np.log(np.abs(fftI)+1.0),cmap='gray')
plt.title('fft de l image sinus simple')

fftIshift=fftshift(fftI) #on translate la fft pour placer le (0,0) fréquentiel au milieu de 

plt.figure(4)
plt.imshow(np.log(np.abs(fftIshift)+1.0),cmap='gray')
plt.title('fft de l image sinus simple apres fftshift')

############################################################################
# Creation d'une image avec 3 fréquences d'amplitudes égales pour deux et une d'amplitude beaucoup plus faible


N=N0
N=625
x=np.arange(0,N,1)
y=np.arange(0,N,1)
X,Y=np.meshgrid(x,y)
tx1=4
ty1=32
tx2=8
ty2=32
tx3=32
ty3=32

X,Y=np.meshgrid(x,y); 
I2=50*np.sin(2*np.pi*tx1/N0*X+2*np.pi*ty1/N0*Y)+50*np.sin(2*np.pi*tx2/N0*X+2*np.pi*ty2/N0*Y)+0.01*np.sin(2*np.pi*tx3/N0*X+2*np.pi*ty3/N0*Y); 

plt.figure(5)
plt.imshow(I2,cmap='gray')

###########################################
#Calcul de la fft et affichage centré
###########################################
fftI=fft2(I2)
fftIshift=fftshift(fftI)
plt.figure(6)
plt.imshow(np.log(np.abs(fftIshift)+1.0),cmap='gray')
plt.title('fft de l image sinus compose')

#multiplication de l'image par une fenêtre de Kaiser

b = 10 ; #force dulissage de la fenêtre
W1=np.reshape(np.kaiser(N,b),(N,1))
W2=np.reshape(np.kaiser(N,b),(1,N))
W3 = W1.dot(W2);


J = I2*W3;
plt.figure(7)
plt.imshow(J,cmap='gray')

fftI=fft2(J)
fftIshift=fftshift(fftI)
plt.figure(8)
plt.imshow(np.log(np.abs(fftIshift)+1.0),cmap='gray')
plt.title('fft de l image 2 fenetree')

