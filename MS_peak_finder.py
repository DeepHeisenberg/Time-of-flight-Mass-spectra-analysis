# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:46:34 2020

@author: y.wei
"""
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
plt.style.use('seaborn-white')
#plt.style.use('ggplot')  its nice!
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['figure.titlesize'] = 25
def peak_finder(mass, dis, h, p):
    
    # mass = pd.read_csv('R5076_41710-v01.csv') 
    # mass.columns= ['a','b','c','d']
    
    
    #plt.figure(figsize=(20, 5))
    #locs, labels = plt.xticks()
    x=mass.iloc[:,0][1000:100000]
    counts= mass.iloc[:,1][1000:100000]
    y= np.log(mass.iloc[:,1])[1000:100000]
    #plt.plot(x,y, color='k')
    #plt.xlim(0,100) 
    
    #mass_my.columns=['m','count']
    #counts = mass_my['count'][0:4000] 
    #log_counts = np.log(counts)
    #x=mass_my['m'][0:4000]
    # yhat = savgol_filter(y, 5, 3) 
    peaks, _ = find_peaks(y, distance=dis, height=h, prominence=p)#5(log_counts, distance=2, height=2, prominence=0.6)
    #peaks = pd.DataFrame(peaks)
    #peaks.to_csv('peaks.csv', index = False)
    fig, ax = plt.subplots(figsize=(20, 5))
    # locs, labels = plt.xticks()
    ax.plot(x,y, color='k')
    ax.scatter(x.iloc[peaks],y.iloc[peaks], color='r',s = 20)
    plt.xlabel('Mass-to-charge ratio [m/z[)]')
    plt.ylabel('Count [log]')
    ax.minorticks_on()
    # plt.tick_params(axis='x',which='minor', bottom=True)
    #plt.title('Mass spectrum at grain boundary')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ax.tick_params(direction='out',axis='both',  which ='major',length=6, width=2, colors='black',
                   grid_color='black', grid_alpha=0.5)
    ax.tick_params(direction='out',axis='both',  which ='minor',length=4, width=2, colors='black')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    
    plt.show()
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # For the minor ticks, use no labels; default NullFormatter.
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.xlim(0,max(x.iloc[peaks]+10))
    plt.ylim(0,15)  
    plt.grid(linestyle='dotted')
    y=y.reset_index(drop=True)
    plt.show()
    new_mass= pd.concat([x.iloc[peaks], counts.iloc[peaks]],axis=1, ignore_index=True, sort =False)
    new_mass.columns = ['m','count']
    return new_mass