# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:57:40 2020

@author: y.wei
"""

import pandas as pd
#from database_creator import ms
import csv
import numpy as np

Ru = pd.DataFrame(data={'m':[96, 98, 99, 100, 101, 102, 104],'Composition':[5.54, 1.87, 12.8, 12.6, 17.1, 31.6, 18.6]})
Mo = pd.DataFrame(data={'m':[92, 94, 95, 96, 97, 98, 100], 'Composition':[14.77, 9.22, 15.90, 16.68, 9.56, 24.20, 9.67]})
Sm = pd.DataFrame(data={'m':[144, 147, 148, 149, 150, 152, 154], 'Composition':[3.07, 14.99, 11.24, 13.82, 7.38, 26.75, 22.75]}) 
print(Sm['m']/3)
Nd = pd.DataFrame(data={'m':[142, 143, 144, 145, 146, 148, 150], 'Composition':[27.2, 12.2, 23.8, 8.30, 17.2, 5.7, 5.6]})

Gd = pd.DataFrame(data={'m':[152, 154, 155, 156, 157, 158, 160], 'Composition':[0.2, 2.18, 14.80, 20.47, 15.65, 24.84, 21.86]})
Yd = pd.DataFrame(data={'m':[168, 170, 171, 172, 173, 174, 176], 'Composition':[0.13, 3.04, 14.28, 21.83, 16.13, 31.83, 12.76]})
    
seven_peaks = [ Ru, Mo, Sm, Nd, Gd, Yd] 
seven_peaks_names = ['Ru', 'Mo', 'Sm', 'Nd', 'Gd', 'Yd'] 
N=5000
delta = 0.01
for peaks, name in zip(seven_peaks, seven_peaks_names):
    n=0
    comps =[]
    print(name)
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
        
        mu6, sigma6 = peaks['Composition'][5], delta*peaks['Composition'][5]
        s6 = np.random.normal(mu6, sigma6, 1) 
        
        mu7, sigma7 = peaks['Composition'][6], delta*peaks['Composition'][6]
        s7 = np.random.normal(mu7, sigma7, 1)         
    #    mu4, sigma4 = 3.635, delta*3.635
    #    s4 = np.random.normal(mu4, sigma4, 1)
    #    mu5, sigma5 = 0.926, delta*0.926
    #    s5 = np.random.normal(mu5, sigma5, 1)
        total =s1[0]+s2[0]+s3[0]+s4[0]+s5[0]+s6[0] +s7[0]
        if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0 and s5>0 and s6>0 and s7>0):
            items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total, 
                   round(s4[0],3)/total, round(s5[0],3)/total, round(s6[0],3)/total, round(s7[0],3)/total]
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
num = 7    
while(n<=N):
    mu1, sigma1 = 100/num, delta* 100/num
    s1 = np.random.normal(mu1, sigma1, 1)
    mu2, sigma2 =  100/num, delta* 100/num
    s2 = np.random.normal(mu2, sigma2, 1)
    mu3, sigma3 =  100/num, delta* 100/num
    s3 = np.random.normal(mu3, sigma3, 1)
    mu4, sigma4 =  100/num, delta* 100/num
    s4 = np.random.normal(mu4, sigma4, 1)
    mu5, sigma5 =  100/num, delta* 100/num
    s5 = np.random.normal(mu5, sigma5, 1)
    
    mu6, sigma6 =  100/num, delta* 100/num
    s6 = np.random.normal(mu6, sigma6, 1)
    
    mu7, sigma7 =  100/num, delta* 100/num
    s7 = np.random.normal(mu7, sigma7, 1)
#    mu4, sigma4 = 3.635, delta*3.635
#    s4 = np.random.normal(mu4, sigma4, 1)
#    mu5, sigma5 = 0.926, delta*0.926
#    s5 = np.random.normal(mu5, sigma5, 1)
    total =s1[0]+s2[0]+s3[0]+s4[0]+s5[0]+s6[0] +s7[0]
    if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 and s4>0 and s5>0 and s6>0 and s7>0):
        items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total, 
               round(s4[0],3)/total, round(s5[0],3)/total, round(s6[0],3)/total, round(s7[0],3)/total]
        comps.append(items)
        n+=1
new_name='Random_class'     
with open(new_name, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(comps)  
    