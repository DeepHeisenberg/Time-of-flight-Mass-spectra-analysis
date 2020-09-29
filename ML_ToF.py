# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:52:11 2020

@author: y.wei
"""

import pandas as pd
import os
import re
import pickle
from atomic_pattern_recognizer import multi_peak_search, two_peak_search, one_peak_search, round_up_values,organic_peak_search

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
from MS_peak_finder import peak_finder
from molecule_pattern_recognizer import molecule_all
from molecule_pattern_recognizer import find_good_molecules
from molecule_pattern_recognizer import molecule_formula
from molecule_pattern_recognizer import molecular_classifier
from elements import H, Be, B, C, N, O, Na, Mg, Al, Si, P, S, K, Ca, Sc, Ti, Cr, Mn, Fe, Fe_3P, Co,Ni
from elements import Cu, Zn, Ga, Ge, As, Se, In, Ag, Zr, Ru, Mo, Sm, Nd, Gd, Yd

test_MS= pd.read_csv('APT/peaks_CuS.csv')

#this part is to round up the m/z ratio to half integer and integer.
MZ_ratio_round = round_up_values((test_MS['m']))
df_test =  pd.DataFrame(MZ_ratio_round)
df_test.columns = ['Da']
frame = pd.concat([df_test, test_MS['count']], axis=1)

#this part identifies all atomic patterns
Identify_atom=False
if Identify_atom == True:
    elements_three_or_more_peak, new_peaks, predictions, name_pair, all_elements_uncertain = multi_peak_search(frame, MZ_ratio_round)
    elements_two_peak = two_peak_search(frame, MZ_ratio_round)
    elements_one_peak = one_peak_search(frame, MZ_ratio_round)
    elements_organic_peak = organic_peak_search(frame, MZ_ratio_round)
    
    all_elements = elements_one_peak + elements_two_peak +  elements_organic_peak + elements_three_or_more_peak      
    all_elements = pd.DataFrame(all_elements)
    all_elements.columns = ['name']
    all_elements = all_elements.name.unique().tolist()
    all_elements = pd.DataFrame(all_elements)
    all_elements.columns = ['name']


#molecular pattern recognizer consist of two parts: searching the molecular formula and classifying them.
Identify_molecule = False

if Identify_molecule == True:
    
    metals=[Cu,In, S]# [Al,Cr,Ti,Co, Ni,Ca] #[Fe, Mn, Si, Al]
    metal_names =['Cu','In','S']#['Al','Cr','Ti','Co', 'Ni','Ca']#['Fe','Mn','Si','Al'] 
    nonmetals = [S, O, H]
    nonmetals_names = ['S','O','H']

    mz_lower_bound = 110
    mz_upper_bound = 115
    peak_of_interest = frame.loc[(frame['Da']>=mz_lower_bound) & (frame['Da']<=mz_upper_bound)]
    
    google_molecules, stats = find_good_molecules(peak_of_interest, metals, metal_names, nonmetals)
    
    
x = molecule_formula(Cu,1,S,2)    
y= molecule_formula(Cu,1,S,1,O,2)
molecules = [x,y]
molecule_classifier =False
if molecule_classifier == True:
    model = molecular_classifier(molecules)

