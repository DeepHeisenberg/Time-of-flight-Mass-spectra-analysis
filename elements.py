# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:50:57 2020

@author: y.wei
"""
import pandas as pd
H = pd.DataFrame(data={'m':[1],'Composition':[100]})
#    Li = pd.DataFrame(data={'m':[6.0, 7.0],'Composition':[7.6, 92.4]}) 
Be = pd.DataFrame(data={'m':[9.0],'Composition':[100]}) 
B = pd.DataFrame(data={'m':[10.0, 11.0],'Composition':[19.9, 80.1]}) 
C = pd.DataFrame(data={'m':[12.0, 13.0],'Composition':[98.89, 1.07]}) 
N = pd.DataFrame(data={'m':[14.0, 15.0],'Composition':[99.64, 0.36]}) 
O = pd.DataFrame(data={'m':[16.0, 18.0],'Composition':[99.8, 0.2]})
#    F = pd.DataFrame(data={'m':[19],'Composition':[100]}) 
Na = pd.DataFrame(data={'m':[23],'Composition':[100]}) 
Mg = pd.DataFrame(data={'m':[24,25,26],'Composition':[78.99,10.00,11.01]}) 
Al = pd.DataFrame(data={'m':[27],'Composition':[100]}) 
Si = pd.DataFrame(data={'m':[28, 29, 30],'Composition':[92.22, 4.69, 3.09]}) 
P = pd.DataFrame(data={'m':[31],'Composition':[100]}) 
S = pd.DataFrame(data={'m':[32.0, 33.0, 34.0, 36.0],'Composition':[94.99, 0.75, 4.25, 0.01]})

K =  pd.DataFrame(data={'m':[39, 40, 41],'Composition':[93.258, 0.0117, 6.730]})
Ca = pd.DataFrame(data={'m':[40, 42, 43, 44, 46, 48],'Composition':[96.94, 0.647, 0.135, 2.086, 0.004, 0.187]}) 
Sc = pd.DataFrame(data={'m':[45],'Composition':[100]}) 

Ti = pd.DataFrame(data={'m':[46.0, 47.0, 48.0, 49.0, 50.0],'Composition':[8.25, 7.44, 73.72, 5.41, 5.18]})
Cr = pd.DataFrame(data={'m':[50.0, 52.0, 53.0, 54.0],'Composition':[4.345, 83.789, 9.501, 2.365]})
Mn = pd.DataFrame(data={'m':[55],'Composition':[100]}) 
Fe = pd.DataFrame(data={'m':[54.0, 56.0, 57.0, 58.0],'Composition':[5.845, 91.754, 2.119, 0.282]})
Fe_3P = pd.DataFrame(data={'m':[54.0, 56.0, 57.0],'Composition':[5.845, 91.754, 2.4]}) 

Co = pd.DataFrame(data={'m':[59],'Composition':[100]})
Ni = pd.DataFrame(data={'m':[58.0, 60.0, 61.0, 62.0, 64.0],'Composition':[68.0, 26.223, 1.140, 3.635, 0.926]})
Cu = pd.DataFrame(data={'m':[63.0, 65.0],'Composition':[69.15, 30.85]})
Zn = pd.DataFrame(data={'m':[64.0, 66.0, 67.0, 68.0, 70.0],'Composition':[48.268, 27.975, 4.102, 19.204, 0.631]})
Ga = pd.DataFrame(data={'m':[69.0, 71.0],'Composition':[60.108, 39.892]})
Ge = pd.DataFrame(data={'m':[70.0, 72.0, 73.0, 74.0, 76.0],'Composition':[20.38, 27.31, 7.76, 36.72, 7.83]})
As = pd.DataFrame(data={'m':[75],'Composition':[100]})
Se = pd.DataFrame(data={'m':[74, 76, 77, 78, 80, 82],'Composition':[0.89, 9.37, 7.64, 23.77, 49.61, 8.73]})
#Se2 = pd.concat([Se['m'].apply(lambda x: x/2), Se['Composition']],axis=1)
In = pd.DataFrame(data={'m':[113.0, 115.0],'Composition':[4.29, 95.71]}) 
Ag = pd.DataFrame(data={'m':[107.0, 109.0],'Composition':[51.839, 48.161]}) 
Zr = pd.DataFrame(data={'m':[90, 91, 92, 94, 96],'Composition':[51.45, 11.22, 17.15, 17.38, 2.80]})

Ru = pd.DataFrame(data={'m':[96, 98, 99, 100, 101, 102, 104],'Composition':[5.54, 1.87, 12.8, 12.6, 17.1, 31.6, 18.6]})
Mo = pd.DataFrame(data={'m':[92, 94, 95, 96, 97, 98, 100], 'Composition':[14.77, 9.22, 15.90, 16.68, 9.56, 24.20, 9.67]})
Sm = pd.DataFrame(data={'m':[144, 147, 148, 149, 150, 152, 154], 'Composition':[3.07, 14.99, 11.24, 13.82, 7.38, 26.75, 22.75]}) 
Nd = pd.DataFrame(data={'m':[142, 143, 144, 145, 146, 148, 150], 'Composition':[27.2, 12.2, 13.8, 8.30, 17.2, 5.7, 5.6]})
Gd = pd.DataFrame(data={'m':[152, 154, 155, 156, 157, 158, 160], 'Composition':[0.2, 2.18, 14.80, 20.47, 15.65, 24.84, 21.86]})
Yd = pd.DataFrame(data={'m':[168, 170, 171, 172, 173, 174, 176], 'Composition':[0.13, 3.04, 14.28, 21.83, 16.13, 31.83, 12.76]})