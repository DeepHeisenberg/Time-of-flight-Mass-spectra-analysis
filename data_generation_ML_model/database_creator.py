# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:14:08 2020

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:24:21 2020

@author: y.wei
"""
import pandas as pd
import numpy as np
#from Simulation_MS import theoretical_mass_spectra
def combine_ions(comp_1, comp_2):
    all_ion = []
    all_compo = []
    total_S = comp_1['m']
    compo_S = comp_1['Composition']
#    start = 0.05
#    end = 1
#    
    total_Cu = comp_2['m']
    compo_Cu = comp_2['Composition']
    for i, s in zip(compo_S, total_S) :
        for j, cu in zip(compo_Cu, total_Cu) :
            ion = s+cu
            compo = round(i*j/100,2)
            all_ion.append(ion)
            all_compo.append(compo)
    CuS =  pd.DataFrame(data={'m':all_ion,'Composition':all_compo})   
    CuS = CuS.groupby('m').sum().reset_index()    
    x = CuS[(CuS!= 0).all(1)].reset_index(drop=True)   
    return x

def ms(x):
    temp =  combine_ions(x[0],x[1])
    if len(x) >=3:
        for i in x[2:]:
            print(i)
            temp = combine_ions(temp, i)
    #        new_temp = temp
        final = temp     
    else:
        final = temp
        
    data = final[final['Composition'] >= 1].reset_index(drop=True)  
    
    CuS =data
    CuS['log_value'] = np.log(CuS.Composition)   
    return CuS
if __name__ == "__main__":
#    H = pd.DataFrame(data={'m':[1],'Composition':[100]})
#    Li = pd.DataFrame(data={'m':[6.0, 7.0],'Composition':[7.6, 92.4]}) 
    Be = pd.DataFrame(data={'m':[9.0],'Composition':[100]}) 
    B = pd.DataFrame(data={'m':[10.0, 11.0],'Composition':[19.9, 80.1]}) 
    C = pd.DataFrame(data={'m':[12.0, 13.0],'Composition':[98.89, 1.07]}) 
    N = pd.DataFrame(data={'m':[14.0, 15.0],'Composition':[99.64, 0.36]}) 
    O = pd.DataFrame(data={'m':[16.0, 18.0],'Composition':[99.8, 0.2]})
#    F = pd.DataFrame(data={'m':[19],'Composition':[100]}) 
    #Na= pd.DataFrame(data={'m':[23],'Composition':[100]}) 
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
    #elements that can combine with other elements and form new compounds!
    double_C=ms([C,C])
    double_S=ms([S,S])
    double_B = ms([B,B])
    #SS=
    #BB=
    #CC=
    
    elements = [Be, B , C, N, O, Mg, Al, Si, P, S,K,Ca, Sc, Ti, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, In, Ag,Zr, double_C, double_S, double_B]
    #elements2 = [H2, Li2, Be2, B2 , C2, N2, O2, F2, Na2, Mg2, Al2, Si2, P2, S2, Ti2, Cr2, Mn2, Fe2, Co2, Ni2, Cu2, Zn2, Ga2, Ge2, As2, Se2]
    elements_c2= []
    names = [ 'Be', 'B' , 'C', 'N', 'O', 'Mg', 'Al', 'Si',
              'P', 'S','K','Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
              'Zn', 'Ga', 'Ge', 'As', 'Se','In','Ag','Zr','double_C','double_S', 'double_B']
    
    names_c2 =  ['Be2', 'B2' , 'C2', 'N2', 'O2', 'Mg2', 'Al2', 'Si2',
              'P2', 'S2', 'K2','Ca2', 'Sc2', 'Ti2', 'Cr2', 'Mn2', 'Fe2', 'Co2', 'Ni2', 'Cu2',
              'Zn2', 'Ga2', 'Ge2', 'As2', 'Se2','In2', 'Ag2','Zr2' ,'double_C2','double_S2', 'double_B2']
    
    len(names)
    for i, element in enumerate(elements):
        element.name = names[i]
        element2 = pd.concat([element['m'].apply(lambda x: x/2), element['Composition']],axis=1)
        elements_c2.append(element2)
        element.to_csv('database/'+names[i]+'.csv',index=False)
        
    for i, element in enumerate(elements_c2):
#        element.name = names_c2[i]
        element.to_csv('database/'+names_c2[i]+'.csv',index=False)
            
        
    #import json
    #with open("my_file", 'w') as outfile:
    #    outfile.write(json.dumps([df.to_dict() for df in elements]))
    #
    #with open("my_file", 'w') as json_file:
    #    data = json.load(json_file)
    #    
    #    element2 = pd.concat([element['m'].apply(lambda x: x/2), element['Composition']],axis=1)
    #    elements_c2.append(element2)
    #    
    
        
    #As = 
    #x=Se
    #x.loc[:,'m'] /= 2
    #Se2['m'] = Se2['m'].apply(lambda x: x/2)]
    #Se2['m'] = Se2['m'].apply(lambda x: x/2)]
    
    #In = pd.DataFrame(data={'m':[113, 115],'Composition':[4.29, 95.71]})