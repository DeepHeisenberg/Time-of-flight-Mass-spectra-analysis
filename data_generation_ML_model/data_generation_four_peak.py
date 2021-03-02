# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:57:40 2020

@author: y.wei
"""

import pandas as pd
#from database_creator import ms
import csv
import numpy as np

Cr = pd.DataFrame(data={'m':[50.0, 52.0, 53.0, 54.0],'Composition':[4.345, 83.789, 9.501, 2.365]})
Zn = pd.DataFrame(data={'m':[64.0, 66.0, 67.0, 68.0],'Composition':[48.268, 27.975, 4.102, 19.204]})
Fe = pd.DataFrame(data={'m':[54.0, 56.0, 57.0, 58.0],'Composition':[5.845, 91.754, 2.119, 0.282]}) 
Ni = pd.DataFrame(data={'m':[58.0, 60.0, 61.0, 62.0],'Composition':[68.0, 26.223, 1.140, 3.635]})
Zr = pd.DataFrame(data={'m':[90, 91, 92, 94],'Composition':[52.45, 11.22, 17.15, 18.38]})# add 1 to first peak and 1 to 4th peak
Ca = pd.DataFrame(data={'m':[40, 42, 44, 48],'Composition':[96.94, 0.647, 2.086,  0.187]}) 


four_peaks = [Cr, Zn, Fe, Ni, Zr,Ca] 
four_peaks_names = ['Cr', 'Zn', 'Fe', 'Ni', 'Zr','Ca'] 
N=5000
delta = 0.01
for peaks, name in zip(four_peaks, four_peaks_names):
    n=0
    print(name)
    comps =[]
    while(n<=N):
        mu1, sigma1 = peaks['Composition'][0], delta*peaks['Composition'][0]
        s1 = np.random.normal(mu1, sigma1, 1)
        mu2, sigma2 = peaks['Composition'][1], delta*peaks['Composition'][1]
        s2 = np.random.normal(mu2, sigma2, 1)
        mu3, sigma3 = peaks['Composition'][2], delta*peaks['Composition'][2]
        s3 = np.random.normal(mu3, sigma3, 1)
        mu4, sigma4 = peaks['Composition'][3], delta*peaks['Composition'][3]
        s4 = np.random.normal(mu4, sigma4, 1)
    #    mu4, sigma4 = 3.635, delta*3.635
    #    s4 = np.random.normal(mu4, sigma4, 1)
    #    mu5, sigma5 = 0.926, delta*0.926
    #    s5 = np.random.normal(mu5, sigma5, 1)
        total =s1[0]+s2[0]+s3[0]+s4[0]
        if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0):
            items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total,  round(s4[0],3)/total]
            comps.append(items)
            n+=1
    new_name='{}'.format(name)     
    with open(new_name, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(comps)  
        
#Mg = np.genfromtxt(new_name, delimiter=',', dtype=None)
#Mg = pd.DataFrame(Mg)
#Mg['std'] = Mg.std(axis=1)
#Mg['Class'] =0
n=0
comps = []        
while(n<=N):
    mu1, sigma1 = 25, delta*25
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 = 25, delta*25
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 25, delta*25
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 25, delta*25
    s4 = np.random.normal(mu4, sigma4, 1)
#    mu4, sigma4 = 3.635, delta*3.635
#    s4 = np.random.normal(mu4, sigma4, 1)
#    mu5, sigma5 = 0.926, delta*0.926
#    s5 = np.random.normal(mu5, sigma5, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]
    if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0):
        items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total,  round(s4[0],3)/total]
        comps.append(items)
        n+=1
new_name='Random_class'     
with open(new_name, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(comps)  
    