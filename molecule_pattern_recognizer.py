# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:10:12 2020

@author: y.wei
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

from matplotlib import pyplot
    

def round_up_values(test):   
    new_test= []
    for n in test:
        temp=int(n)
        if (n>=temp+0.35 and n<=temp+0.55):
            new_n = temp+0.5
        if (n>=temp-0.35 and n<=temp+0.35):
            new_n = temp        
        if  n>=temp+0.55:
            new_n = temp+1
        new_test.append(new_n)        
    return new_test
        

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



def molecule_formula(*args):
    # print(args)
    temp =  args[0]
    if args[1] >=2:
        for _ in range(1, args[1]):
            temp = combine_ions(temp, args[0])
    #        new_temp = temp
    for i in range(2, len(args), 2):
        # print(i)
        # print(len(args))
        # print(args[i])
        if args[i+1]>0:        
            for _ in range(args[i+1]):
                temp = combine_ions(temp, args[i])    
                
    final=temp[temp['Composition']>1]
    return final
# molecule = molecule_temp(Al, 2)
def molecule_all(*args):
    all_molecule = []
    stats = []
    for charge in range(1,4):
        if len(args[1]) == 1:
            for n in range(1,4):  
                for m in range(0,4):
                    molecule = molecule_formula(args[0], n, args[1][0], m)
                    molecule['m'] /= charge
                    all_molecule.append(molecule)
                    # print(molecule)
                    stat = [n, m, charge]
                    stats.append(stat)

        if len(args[1]) == 2:
            for n in range(1, 4):  
                for m in range(0,4):
                    for a in range(0,4):
                        molecule = molecule_formula(args[0], n, args[1][0], m,  args[1][1], a)
                        molecule['m'] /= charge
                        all_molecule.append(molecule)
                        # print(molecule)
                        stat = [n, m, a, charge]
                        stats.append(stat)    
                        
        if len(args[1]) == 3:
            for n in range(1, 4):  
                for m in range(0, 4):
                    for a in range(0, 4):
                        for b in range(0, 4):
                            molecule = molecule_formula(args[0], n, args[1][0], m,  args[1][1], a, args[1][2], b)
                            molecule['m'] /= charge
                            all_molecule.append(molecule)
                            # print(molecule)
                            stat = [n, m, a, b, charge]
                            stats.append(stat)            
    stats= pd.DataFrame(stats)     
    return all_molecule, stats


Fe = pd.DataFrame(data={'m':[54.0, 56.0, 57.0, 58.0],'Composition':[5.845, 91.754, 2.119, 0.282]})

def find_good_molecules(frame, metals, metals_names, nonmetals):  
    all_molecule = []
    stats=[]
    N_metal = []
    total = []
    good_molecules = [] 
    metals_name_dev=[]
    normalized_Is = []
    MZ_ratio_round = frame['Da'].tolist()
    for N, metal in enumerate(metals):
        all_molecule, stats = molecule_all(metal, nonmetals)       
        for i, molecule in enumerate(all_molecule):            
            m = molecule['m'].isin(MZ_ratio_round).tolist()
            index=np.where(m)[0]
            comp_molecule = np.array(molecule['Composition'])[index]
            x = frame[frame['Da'].isin(molecule['m'][m])]
            normalized_I=x['count']/sum(x['count'])*100        
            x['normalized_count'] = normalized_I
            deviation = sum(abs(normalized_I-comp_molecule))
            
            if (sum(comp_molecule) >= 98 and len(comp_molecule) > 0 and deviation <= 20):
                if (sum(stats.iloc[i][1:])<=10 and sum(stats.iloc[i][1:])>=2):
                    total.append(stats.iloc[i])
                    metals_name_dev.append([metals_names[N],deviation])
                    N_metal.append(N)
                    good_molecules.append(molecule)
                    normalized_Is.append(x)

    total = pd.DataFrame(total)
    total['metal_names_and_dev'] = metals_name_dev   
    total['N_metal'] = N_metal           

    return good_molecules, total

def molecular_classifier(total):
    item = total[0]
    frame=[]
    N=1000
    comps =[]
    delta =0.01
    for n,item in enumerate(total):
        augmented_class = []
        for i in range(N):
            # print(i)
            comps=[]
            for j in range(len(item['Composition'])):
                mu, sigma = item['Composition'][j], delta*item['Composition'][j]
                s = np.random.normal(mu, sigma, 1)
                comps.append(s[0])
                total = sum(comps)
                if (total<=(sum(item.Composition)+1) and total>=(sum(item.Composition)-1)):
                    augmented_class.append(comps)
        augmented_class= pd.DataFrame(augmented_class)  
        augmented_class['Class'] = n          
        frame.append(augmented_class)        
    data =  pd.concat(frame)             
    X_data = data.drop('Class',axis=1)
    y = data.Class 
    
    print('training test spilting...')
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)
    print('start_training...')
    evals_result = {}
    params = {
              "objective" : "multiclass",
              "num_class" : len(frame),
              "num_leaves" : 30,
              "max_depth": 5,
              "learning_rate" : 0.01,
              "bagging_fraction" : 0.9,  # subsample
              "feature_fraction" : 0.9,  # colsample_bytree
              "bagging_freq" : 5,        # subsample_freq
              "bagging_seed" : 20,
              "verbosity" : -1 ,
              'metric':'multi_logloss'}
    lgtrain, lgval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)
    
    model_lgb = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], evals_result=evals_result,
                          early_stopping_rounds=100, verbose_eval=200)
    # print('Plotting metrics recorded during training...')
    # lgb.plot_metric(evals_result, metric='multi_logloss')
    # plt.xlim(0,1000) 
    # plt.show()
    return model_lgb
