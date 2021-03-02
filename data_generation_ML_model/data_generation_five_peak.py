# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:57:40 2020

@author: y.wei
"""

import pandas as pd
#from database_creator import ms
import csv
import numpy as np

Ti = pd.DataFrame(data={'m':[46.0, 47.0, 48.0, 49.0, 50.0],'Composition':[8.25, 7.44, 73.72, 5.41, 5.18]})
Ni = pd.DataFrame(data={'m':[58.0, 60.0, 61.0, 62.0, 64.0],'Composition':[68.0, 26.223, 1.140, 3.635, 0.926]})
Ge = pd.DataFrame(data={'m':[70.0, 72.0, 73.0, 74.0, 76.0],'Composition':[20.38, 27.31, 7.76, 36.72, 7.83]})
Zn = pd.DataFrame(data={'m':[64.0, 66.0, 67.0, 68.0, 70.0],'Composition':[48.268, 27.975, 4.102, 19.204, 0.631]})
Se = pd.DataFrame(data={'m':[76, 77, 78, 80, 82],'Composition':[9.37, 7.64, 23.77, 49.61, 8.73]})
Zr = pd.DataFrame(data={'m':[90, 91, 92, 94, 96],'Composition':[51.45, 11.22, 17.15, 17.38, 2.80]})

five_peaks = [Ti, Ni, Ge, Zn, Se, Zr] 
five_peaks_names = ['Ti', 'Ni', 'Ge','Zn','Se','Zr'] 
N=5000
delta = 0.01
for peaks, name in zip(five_peaks, five_peaks_names):
    n=0
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
        mu5, sigma5 = peaks['Composition'][4], delta*peaks['Composition'][4]
        s5 = np.random.normal(mu5, sigma5, 1)        
    #    mu4, sigma4 = 3.635, delta*3.635
    #    s4 = np.random.normal(mu4, sigma4, 1)
    #    mu5, sigma5 = 0.926, delta*0.926
    #    s5 = np.random.normal(mu5, sigma5, 1)
        total =s1[0]+s2[0]+s3[0]+s4[0]+s5[0]
        if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0 and s5>0):
            items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total,  round(s4[0],3)/total, round(s5[0],3)/total]
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
    mu1, sigma1 = 20, delta*20
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 = 20, delta*20
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 = 20, delta*20
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 = 20, delta*20
    s4 = np.random.normal(mu4, sigma4, 1)
    mu5, sigma5 = 20, delta*20
    s5 = np.random.normal(mu5, sigma5, 1)
#    mu4, sigma4 = 3.635, delta*3.635
#    s4 = np.random.normal(mu4, sigma4, 1)
#    mu5, sigma5 = 0.926, delta*0.926
#    s5 = np.random.normal(mu5, sigma5, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]+s5[0]
    if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0 and s5>0):
        items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total,  round(s4[0],3)/total, round(s5[0],3)/total]
        comps.append(items)
        n+=1
new_name='Random_class'     
with open(new_name, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(comps)  
    