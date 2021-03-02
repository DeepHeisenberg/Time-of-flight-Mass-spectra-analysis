# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:57:40 2020

@author: y.wei
"""

import pandas as pd
from database_creator import ms
import csv
import numpy as np

B = pd.DataFrame(data={'m':[10.0, 11.0],'Composition':[19.9, 80.1]}) 
S = pd.DataFrame(data={'m':[32.0, 33.0, 34.0, 36.0],'Composition':[94.99, 0.75, 4.25, 0.01]})
Mg = pd.DataFrame(data={'m':[24,25,26],'Composition':[78.99,10.00,11.01]}) 
Si = pd.DataFrame(data={'m':[28, 29, 30],'Composition':[92.22, 4.69, 3.09]}) 
Fe = pd.DataFrame(data={'m':[54.0, 56.0, 57.0, 58.0],'Composition':[5.845, 91.754, 2.119, 0.282]}) 
Ca = pd.DataFrame(data={'m':[40, 42, 44],'Composition':[96.94, 0.647, 2.086]}) 
Ni_3P = pd.DataFrame(data={'m':[58.0, 60.0, 62.0],'Composition':[69.0, 27.223, 3.635]})
Cr_3P = pd.DataFrame(data={'m':[50.0, 52.0, 53.0],'Composition':[4.8, 84.789, 10.001]})
double_B = ms([B,B])
double_S = ms([S,S])
three_peaks = [S, Mg, Si, Fe, Ca, double_B, double_S, Ni_3P, Cr_3P] 
three_peaks_names = ['S', 'Mg', 'Si', 'Fe','Ca', 'double_B','double_S','Ni_3P','Cr_3P'] 
N=5000
delta = 0.01
for peaks, name in zip(three_peaks, three_peaks_names):
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
    #    mu4, sigma4 = 3.635, delta*3.635
    #    s4 = np.random.normal(mu4, sigma4, 1)
    #    mu5, sigma5 = 0.926, delta*0.926
    #    s5 = np.random.normal(mu5, sigma5, 1)
        total =s1[0]+s2[0]+s3[0]
        if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0):
            items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total]
            comps.append(items)
            n+=1
    new_name='{}'.format(name)     
    with open(new_name, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(comps)  
        
# n=0
# comps = []        
# while(n<=N):
#     mu1, sigma1 = 33.3, delta*33.3
#     s1 = np.random.normal(mu1, sigma1, 1)
#     mu2, sigma2 = 33.3, delta*33.3
#     s2 = np.random.normal(mu2, sigma2, 1)
#     mu3, sigma3 = 33.3, delta*33.3
#     s3 = np.random.normal(mu3, sigma3, 1)

# #    mu4, sigma4 = 3.635, delta*3.635
# #    s4 = np.random.normal(mu4, sigma4, 1)
# #    mu5, sigma5 = 0.926, delta*0.926
# #    s5 = np.random.normal(mu5, sigma5, 1)
#     total =s1[0]+s2[0]+s3[0]
#     if (total<=101 and total>=99 and s1>0 and s2>0 and s3>0 ):
#         items=[round(s1[0],3)/total, round(s2[0],3)/total, round(s3[0],3)/total]
#         comps.append(items)
#         n+=1
# new_name='Random_class'     
# with open(new_name, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     writer.writerows(comps)  
