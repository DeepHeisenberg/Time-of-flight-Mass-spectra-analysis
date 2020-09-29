# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:52:59 2020

@author: y.wei
"""
import pandas as pd
import os
import re
import numpy as np
import pickle
def round_up_values(test):
    new_test= []
    for n in test:
        print(n)
        temp=int(n)
#        print(temp)
        if (n>=temp+0.25 and n<temp+0.66):
            new_n = temp+0.5
            new_test.append(new_n)   
        elif  n>=temp+0.66:
            new_n = temp+1
            new_test.append(new_n)  
        else:
            new_n = temp
            new_test.append(new_n)  
        print(new_n)
    return new_test


three_peaks_names = ['Mg', 'S', 'Si', 'Fe', 'Ca_3P','double_S', 'double_B','Random_class_three', 'Ni_3P','Cr_3P']
four_peaks_names = ['Cr', 'Zn', 'Fe', 'Ni', 'Zr','Ca','Random_class_four']
five_peaks_names = ['Ti', 'Ni', 'Ge','Zn','Se','Zr','Random_class_five'] 
seven_peaks_names = ['Ru', 'Mo', 'Sm', 'Nd', 'Gd', 'Yd','Random_class_seven']
three_peak_classifier = pickle.load(open("ML_models/lgb_model_three.pickle.dat", "rb"))
four_peak_classifier = pickle.load(open("ML_models/lgb_model_four.dat", "rb"))
five_peak_classifier = pickle.load(open("ML_models/lgb_model_five.pickle.dat", "rb"))
seven_peak_classifier = pickle.load(open("ML_models/lgb_model_seven.pickle.dat", "rb"))


def three_peak_class(peak):
    x = {'P1': [peak['normalized_count'].iloc[0]], 'P2': [peak['normalized_count'].iloc[1]],
         'P3':[peak['normalized_count'].iloc[2]]}
    x1=pd.DataFrame(x)
    pred = three_peak_classifier.predict(x1)
    name = three_peaks_names[np.argmax(pred[0])]
    x1['std'] =x1.std(axis=1)

    return pred,name, x1 


def four_peak_class(peak):
    x = {'P1': [peak['normalized_count'].iloc[0]], 'P2': [peak['normalized_count'].iloc[1]],
         'P3':[peak['normalized_count'].iloc[2]],'P4':[peak['normalized_count'].iloc[3]]}
    x1 = pd.DataFrame(x)
    x1['std'] =x1.std(axis=1)
    x1['P1/P2'] = x1['P1']/x1['P2']
    x1['P1/P3'] = x1['P1']/x1['P3']
    x1['P1/P4'] = x1['P1']/x1['P4']
    x1['P2/P3'] = x1['P2']/x1['P3']
    x1['P2/P4'] = x1['P2']/x1['P4']
    x1['P3/P4'] = x1['P3']/x1['P4']
    x2 = x1.drop(['P1','P2','P3','P4'],axis=1)
    pred = four_peak_classifier.predict(x2)
    name = four_peaks_names[np.argmax(pred[0])]
    return pred, name, x1    


def five_peak_class(peak):
    x = {'P1': [peak['normalized_count'].iloc[0]], 'P2': [peak['normalized_count'].iloc[1]],
         'P3':[peak['normalized_count'].iloc[2]],'P4':[peak['normalized_count'].iloc[3]],
         'P5':[peak['normalized_count'].iloc[4]]}
    x1=pd.DataFrame(x)
    x1['std'] =x1.std(axis=1)
    pred = five_peak_classifier.predict(x1)
    name = five_peaks_names[np.argmax(pred[0])]
    return pred,name ,x1      


def seven_peak_class(peak):
    x = {'P1': [peak['normalized_count'].iloc[0]], 'P2': [peak['normalized_count'].iloc[1]],
         'P3':[peak['normalized_count'].iloc[2]],'P4':[peak['normalized_count'].iloc[3]],
         'P5':[peak['normalized_count'].iloc[4]], 'P6':[peak['normalized_count'].iloc[5]],'P7':[peak['normalized_count'].iloc[6]]}
                                                      
    x1=pd.DataFrame(x)
    x1['std'] =x1.std(axis=1)
    pred = seven_peak_classifier.predict(x1)
    name = seven_peaks_names[np.argmax(pred[0])]
    return pred,name ,x1  


def MZ_ratio_match(frame, new_test):
    path = 'atomic_database'
    database = []
    database_std = []
    database_name = []
    for filename in os.listdir(path):
    #    print(filename)
        temp = pd.read_csv(path+'/'+filename)
        m = temp['m'].isin(new_test)
        cols = temp.index[m].tolist()
        comp = temp['Composition'].iloc[cols].tolist()
        # print(comp)
        if (len(cols)>=3 and sum(comp)>=99 and len(comp)>=3):
    #        print(x)
    #        print(frame[frame['Da'].isin(temp['m'][m])])
    
            x=frame[frame['Da'].isin(temp['m'][m])]
            normalized_I=x['count']/sum(x['count'])*100        
            x['normalized_count'] = normalized_I 
            # print(x)
            # print(filename)
            std_comp =  temp['Composition'].std()
            std_temp = x['normalized_count'].std()        
            database.append(x)
            database_std.append(std_comp)
    #        print(m)
    #        print(len(m))
            fn = filename.split('.')
            database_name.append(fn[0])
    return database,database_std, database_name 


def multi_peak_search(frame,new_test):    
    database, database_std, database_name =  MZ_ratio_match(frame, new_test)       
    name_pair =  []       
    predictions = []
    new_peaks = []       
    name_pair_uncertain =[]
    new_peaks_uncertain = []
    predictions_uncertain = [] 
    
    for name, peak, std in zip(database_name, database, database_std):   
        # print(name)
        if len(peak) == 3:
            # print(peak)
            prediction, pred_name, x  = three_peak_class(peak)
            print(pred_name)
            print(prediction)
            std_deviation = abs(x['std'][0]-std)/std
            if max(prediction[0])<=0.8:
                x_uncertain = sorted(zip(prediction[0], three_peaks_names), reverse=True)[:3]
                x__uncertain_pred = sorted(prediction[0], reverse=True)[:3]                
            if std_deviation <= 0.5 and max(prediction[0])>=0.8:
                print(peak)
                print(pred_name)
                print('std is {}'.format(std))
                print('deviation from std is {}'.format(std_deviation))
                name_pair.append([pred_name,name]) 
                predictions.append(prediction)
                new_peaks.append(peak)
            elif(std_deviation <= 0.5 and max(prediction[0])<=0.8):
                print(x)
                print('std is {}'.format(std))
                print('deviation from std is {}'.format(std_deviation))
                name_pair_uncertain.append([x_uncertain[0][1],x_uncertain[1][1], x_uncertain[2][1], name]) 
                predictions_uncertain.append(prediction)
                new_peaks_uncertain.append(peak)                  
                
        if len(peak) == 4:
            prediction, pred_name, x  = four_peak_class(peak)
            std_deviation = abs(x['std'][0]-std)/std
            if max(prediction[0])<=0.8:
                x_uncertain = sorted(zip(prediction[0], four_peaks_names), reverse=True)[:3]
                x__uncertain_pred = sorted(prediction[0], reverse=True)[:3]                

            if (std_deviation <= 0.5 and max(prediction[0])>=0.8):
                print(peak)
                print(pred_name)
                # print('std is {}'.format(std))
                # print('deviation from std is {}'.format(std_deviation))
                name_pair.append([pred_name,name]) 
                predictions.append(prediction)
                new_peaks.append(peak)
            elif(std_deviation <= 0.5 and max(prediction[0])<=0.5):
                print(x)
    #            print(peak)
                print(pred_name)
                print('std is {}'.format(std))
                print('deviation from std is {}'.format(std_deviation))
                name_pair_uncertain.append([x_uncertain[0][1],x_uncertain[1][1], x_uncertain[2][1], name]) 
                predictions_uncertain.append(prediction)
                new_peaks_uncertain.append(peak)            
                
        if len(peak) == 5:
            prediction, pred_name, x  = five_peak_class(peak)
            print(pred_name)
            std_deviation = abs(x['std'][0]-std)/std
            print('std is {}'.format(std))
            print('deviation from std is {}'.format(std_deviation))
            if std_deviation <= 0.5:
                name_pair.append([pred_name,name]) 
                predictions.append(prediction)
                new_peaks.append(peak)
                                
        if len(peak) == 7:
            prediction, pred_name, x  = seven_peak_class(peak)
    #        print(pred_name)
            std_deviation = abs(x['std'][0]-std)/std
            print('std is {}'.format(std))
            print('deviation from std is {}'.format(std_deviation))
            if std_deviation <= 0.5:
                name_pair.append([pred_name,name]) 
                predictions.append(prediction)
                new_peaks.append(peak)
                
                
    elements_three_or_more_peak=[]
    for item in name_pair:
        if item[0] == item[1]:
            print('Element {} is confirmed!'.format(item[0]))
            elements_three_or_more_peak.append(item[1])
        if item[0] in item[1]:
                print('Element {} is confirmed!'.format(item[1]))
                elements_three_or_more_peak.append(item[1])
    if len(name_pair_uncertain)>0:
        print('there are peaks with uncertainies!')
        all_elements_uncertain = [name_pair_uncertain, new_peaks_uncertain, predictions_uncertain]            
    else:
        all_elements_uncertain = []
    return elements_three_or_more_peak, new_peaks, predictions, name_pair, all_elements_uncertain


def two_peak_search(frame, new_test):
    path = 'atomic_database'
    database =[] 
    database_std =[]
    database_name=[]
    total_count = sum(frame['count'])
    for filename in os.listdir(path):
    #    print(filename)
        temp = pd.read_csv(path+'/'+filename)
        m = temp['m'].isin(new_test)
        cols = temp.index[m].tolist()
        comp = temp['Composition'].iloc[cols].tolist()
    #    print(m)
        std_comp =  temp['Composition'].std()
        x=frame[frame['Da'].isin(temp['m'][m])]
        if (len(cols)==2 and sum(comp)>=98 and len(x)>0):
            print(frame[frame['Da'].isin(temp['m'][m])])
            normalized_I=x['count']/sum(x['count'])*100        
            x['normalized_count'] = normalized_I
            if ((sum(x['count'])/total_count)>=0):
                std = x['normalized_count'].std()
                std_deviation = abs(std_comp-std)/std_comp
                if  std_deviation<=0.5:
                    database.append(x)
                    database_std.append(std_deviation)
                    fn = filename.split('.')
                    database_name.append(fn[0])
                    
                    print(fn)
    
    elements_two_peak = database_name
    return elements_two_peak


def one_peak_search(frame, new_test):
    path = 'atomic_database'
    database =[] 
    database_std =[]
    database_name=[]
    total_count = sum(frame['count'])
    for filename in os.listdir(path):
    #    print(filename)
        temp = pd.read_csv(path+'/'+filename)
        m = temp['m'].isin(new_test)
        cols = temp.index[m].tolist()
        comp = temp['Composition'].iloc[cols].tolist()
        std_comp =  temp['Composition'].std()
        x=frame[frame['Da'].isin(temp['m'][m])]
        if (len(cols)==1 and sum(comp)>=98 and len(x)>0):
            print(comp)
            normalized_I=x['count']/sum(x['count'])*100        
            x['normalized_count'] = normalized_I
            if ((sum(x['count'])/total_count)>=0.0001):
                std_deviation = 0
                database.append(x)
                database_std.append(std_deviation)
                fn = filename.split('.')
                database_name.append(fn[0])
    elements_one_peak = database_name
    return elements_one_peak


def organic_peak_search(frame,new_test):
    database =[] 
    database_std =[]
    database_name=[]
    path = 'organic_database'
    elements_organic_peak = []
    for filename in os.listdir(path):
    #    print(filename)
        temp = pd.read_csv(path+'/'+filename)
        m = temp['m'].isin(new_test)
        cols = temp.index[m].tolist()
        comp = temp['Composition'].iloc[cols].tolist()
    
        if (len(cols)<=3 and sum(comp)>=98):
            print(frame[frame['Da'].isin(temp['m'][m])])
            std_comp =  temp['Composition'].std()
            x=frame[frame['Da'].isin(temp['m'][m])]
            normalized_I=x['count']/sum(x['count'])*100        
            x['normalized_count'] = normalized_I 
            database.append(x)
            fn = filename.split('.')
            elements_organic_peak.append(fn[0])
            print(fn)
    return elements_organic_peak

