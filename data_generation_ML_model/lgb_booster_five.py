# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:18:25 2020

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:42:22 2019

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:08:39 2019

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:43:11 2019

@author: y.wei
"""
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

from matplotlib import pyplot

Ti = np.genfromtxt('Ti', delimiter=',', dtype=None)
Ti*=100
Ti = pd.DataFrame(Ti)
Ti['std'] = Ti.std(axis=1)
Ti['Class'] = 0

Ni = np.genfromtxt('Ni', delimiter=',', dtype=None)
Ni *=100
Ni = pd.DataFrame(Ni)
Ni['std'] = Ni.std(axis=1)
Ni['Class'] = 1

Ge = np.genfromtxt('Ge', delimiter=',', dtype=None)
Ge *=100
Ge = pd.DataFrame(Ge)
Ge['std'] = Ge.std(axis=1)
Ge['Class'] = 2


Zn = np.genfromtxt('Zn', delimiter=',', dtype=None)
Zn *=100
Zn = pd.DataFrame(Zn)
Zn['std'] = Zn.std(axis=1)
Zn['Class'] = 3

Se = np.genfromtxt('Se', delimiter=',', dtype=None)
Se *=100
Se = pd.DataFrame(Se)
Se['std'] = Se.std(axis=1)
Se['Class'] = 4

Zr = np.genfromtxt('Zr', delimiter=',', dtype=None)
Zr *=100
Zr = pd.DataFrame(Zr)
Zr['std'] = Zr.std(axis=1)
Zr['Class'] = 5

Random_class = np.genfromtxt('Random_class', delimiter=',', dtype=None)
Random_class *= 100
Random_class = pd.DataFrame(Random_class)
Random_class['std'] = Random_class.std(axis=1)
Random_class['Class'] = 6

#Cu2S5 = pd.read_csv('Cu2S5_comp_augmentation.csv',delimiter=',', dtype=None)
#Cu2S5.columns =['P1/P2','P1/P3','P2/P3']
#Cu2S5['Class'] =11

#Random['maxmin'] = Random.max(axis=1)/Cr.min(axis=1)

frame = [Ti,Ni, Ge, Zn, Se,Zr, Random_class]
data=  pd.concat(frame)


X_data = data.drop('Class',axis=1)
y=data.Class


print('training test spilting...')
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)

#model1 = xgb.XGBClassifier()
#model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
#
#train_model1 = model1.fit(X_trian, y_train)
#train_model2 = model2.fit(X_trian, y_train)

#from sklearn.metrics import classification_report

#pred1 = train_model1.predict_proba(X_test)
#pred2 = train_model2.predict_proba(X_test)
#
#print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))
#print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))
#
#from sklearn.metrics import accuracy_score
#
#print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
#print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))
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
print('Plotting metrics recorded during training...')
lgb.plot_metric(evals_result, metric='multi_logloss')
plt.xlim(0,1000) 
plt.show()
plt.savefig('{}.png'.format('histo'), dpi=800, bbox_inches='tight')        
#from sklearn.metrics import accuracy_score
#train_model3 = model_lgb.fit(X_train, y_train)
pred3 = model_lgb.predict(X_test)
prediction = []
for pred in pred3:
#    print(pred)
    index = np.argmax(pred)
    prediction.append(index)
#predictions_lgbm_ = np.argmax(pred3[0][:])

#predictions_lgbm_01 = np.where(pred3 > 0.5, 1, 0)

#print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, predictions_lgbm_01) * 100))



import itertools

from sklearn.metrics import confusion_matrix




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    
#class_names = ['Ti','Cr', 'Se', 'Zn', 'Se', 'Zr','Random_class']
#
#cnf_matrix = confusion_matrix(y_test, prediction)
#plt.style.use('seaborn-white')
##plt.style.use('ggplot')  its nice!
#plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#plt.rcParams['font.size'] = 16
#plt.rcParams['axes.labelsize'] = 20
#plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams['axes.titlesize'] = 20
#plt.rcParams['xtick.labelsize'] = 18
#plt.rcParams['ytick.labelsize'] = 18
#plt.rcParams['legend.fontsize'] = 16
#plt.rcParams['figure.titlesize'] = 12
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix')
#title = 'Five'     
#plt.savefig('{}.png'.format(title), dpi=800, bbox_inches='tight')     
#
#import pickle
#pickle.dump(model_lgb, open("lgb_model_five.pickle.dat", "wb"))

#x=pd.DataFrame({'0': [44.268, 12.975, 21.102, 17.024, 0.431]}).T
#x['var'] = x.var(axis=1)
#x['maxmin'] = x.max(axis=1)/x.min(axis=1)
#pred = train_model3.predict_proba(x)


#x1=pd.DataFrame({'0': [15.845, 81.754, 0.119, 2.282]}).T
#x1['var'] = x1.var(axis=1)
##x1['maxmin'] = x1.max(axis=1)/x.min(axis=1)
#pred1 = train_model3.predict_proba(x1)
#p=pd.DataFrame({'0': [5.845, 91.754, 2.119, 0.282]}).T 
#new_x1=p*81.754/91.754
#
#
#x2=pd.DataFrame({'0': [4.345, 83.789, 9.501, 1.365]}).T
#x2['var'] = x2.var(axis=1)
#x2['maxmin'] = x2.max(axis=1)/x2.min(axis=1)
#pred2 = train_model3.predict_proba(x2)
#1:[4.345, 83.789, 9.501, 2.365]
#2: [5.845, 91.754, 2.119, 0.282]

#from lgb import plot_importance
lgb.plot_importance(model_lgb)
pyplot.show()