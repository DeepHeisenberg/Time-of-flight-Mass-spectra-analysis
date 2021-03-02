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

Mg = np.genfromtxt('Mg', delimiter=',', dtype=None)
Mg*=100
Mg = pd.DataFrame(Mg)
# Mg.columns=['First Peak','Second Peak','Third Peak']
# Mg['std'] = Mg.std(axis=1)
# Mg['P1/P2'] = Mg[0]/Mg[1]
# Mg['P1/P3'] = Mg[0]/Mg[2]
# Mg['P2/P3'] = Mg[1]/Mg[2]
# Mg = Mg.drop([0,1,2],axis=1)
Mg['Class'] = 0

S = np.genfromtxt('S', delimiter=',', dtype=None)
S *= 100
S = pd.DataFrame(S)
# S.columns=['First Peak','Second Peak','Third Peak']

# S['std'] = S.std(axis=1)
# S['P1/P2'] = S[0]/S[1]
# S['P1/P3'] = S[0]/S[2]
# S['P2/P3'] = S[1]/S[2]
# S = S.drop([0,1,2],axis=1)
S['Class'] = 1

Si = np.genfromtxt('Si', delimiter=',', dtype=None)
Si *= 100
Si = pd.DataFrame(Si)
S.columns=['First Peak','Second Peak','Third Peak']

# Si['std'] = Si.std(axis=1)
# Si['P1/P2'] = Si[0]/Si[1]
# Si['P1/P3'] = Si[0]/Si[2]
# Si['P2/P3'] = Si[1]/Si[2]
# Si = Si.drop([0,1,2],axis=1)
Si['Class'] = 2

Fe = np.genfromtxt('Fe', delimiter=',', dtype=None)
Fe *= 100
Fe = pd.DataFrame(Fe)
# Fe['std'] = Fe.std(axis=1)
# Fe['P1/P2'] = Fe[0]/Fe[1]
# Fe['P1/P3'] = Fe[0]/Fe[2]
# Fe['P2/P3'] = Fe[1]/Fe[2]
# Fe = Fe.drop([0,1,2],axis=1)
Fe['Class'] = 3


Ca = np.genfromtxt('Ca', delimiter=',', dtype=None)
Ca *= 100
Ca = pd.DataFrame(Ca)
# Ca['std'] = Ca.std(axis=1)
# Ca['P1/P2'] = Ca[0]/Ca[1]
# Ca['P1/P3'] = Ca[0]/Ca[2]
# Ca['P2/P3'] = Ca[1]/Ca[2]
# Ca = Ca.drop([0,1,2],axis=1)
Ca['Class'] = 4


double_S = np.genfromtxt('double_S', delimiter=',', dtype=None)
double_S *= 100 
double_S  = pd.DataFrame(double_S)
# double_S['std'] = double_S.std(axis=1)
# double_S['P1/P2'] = double_S[0]/double_S[1]
# double_S['P1/P3'] = double_S[0]/double_S[2]
# double_S['P2/P3'] = double_S[1]/double_S[2]
# double_S = double_S.drop([0,1,2],axis=1)
double_S['Class'] = 5

double_B = np.genfromtxt('double_B', delimiter=',', dtype=None)
double_B *=100
double_B  = pd.DataFrame(double_B)
# double_B['std'] = double_B.std(axis=1)
# double_B['P1/P2'] = double_B[0]/double_B[1]
# double_B['P1/P3'] = double_B[0]/double_B[2]
# double_B['P2/P3'] = double_B[1]/double_B[2]
# double_B = double_B.drop([0,1,2],axis=1)
double_B['Class'] = 6



Random_class = np.genfromtxt('Random_class', delimiter=',', dtype=None)
Random_class *= 100
Random_class = pd.DataFrame(Random_class)
# Random_class['std'] = Random_class.std(axis=1)
# Random_class['P1/P2'] = Random_class[0]/Random_class[1]
# Random_class['P1/P3'] = Random_class[0]/Random_class[2]
# Random_class['P2/P3'] = Random_class[1]/Random_class[2]
# Random_class = Random_class.drop([0,1,2],axis=1)
Random_class['Class'] = 7

Ni = np.genfromtxt('Ni_3P', delimiter=',', dtype=None)
Ni *= 100
Ni = pd.DataFrame(Ni)
# Ni['std'] = Ni.std(axis=1)
# Ni['P1/P2'] = Ni[0]/Ni[1]
# Ni['P1/P3'] = Ni[0]/Ni[2]
# Ni['P2/P3'] = Ni[1]/Ni[2]
# Ni = Ni.drop([0,1,2],axis=1)
Ni['Class'] = 8
#Random['maxmin'] = Random.max(axis=1)/Cr.min(axis=1)
Cr = np.genfromtxt('Cr_3P', delimiter=',', dtype=None)
Cr *= 100
Cr = pd.DataFrame(Cr)
# Cr['std'] = Cr.std(axis=1)
# Cr['P1/P2'] = Cr[0]/Cr[1]
# Cr['P1/P3'] = Cr[0]/Cr[2]
# Cr['P2/P3'] = Cr[1]/Cr[2]
# Cr = Cr.drop([0,1,2],axis=1)
Cr['Class'] = 9
frame = [Mg, S, Si, Fe, Ca, double_S, double_B, Random_class, Ni,Cr]
data=  pd.concat(frame)


X_data = data.drop('Class',axis=1)
y=data.Class


print('training test spilting...')
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=test_size, random_state=seed)

print('start_training...')
evals_result = {}
#weights = [1/6,1/6,1/6, 1/2]
params = {
          "objective" : "multiclass",
          "num_class" : len(frame),
          "num_leaves" : 50,
          'min_data_in_leaf': 20,
          "max_depth": 10,
          'n_estimator': 500,
          "learning_rate" : 0.005,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 20,
           # 'lambda_l1':  0.2,
           # 'lambda_l2': 1,
          "verbosity" : -1 ,
          'metric':'multi_logloss'}
lgtrain, lgval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test)

model_lgb = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval],
                      evals_result=evals_result, early_stopping_rounds=100, verbose_eval=200)
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
import pickle
pickle.dump(model_lgb, open("lgb_model_three.pickle.dat", "wb"))


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

    
class_names = ['Mg','S','Si', 'Fe','Ca','double_S','double_B','Random_Class', 'Ni','Cr']

cnf_matrix = confusion_matrix(y_test, prediction)
plt.style.use('seaborn-white')
#plt.style.use('ggplot')  its nice!
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.titlesize'] = 12
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig('{}.png'.format('confu_three'), dpi=800, bbox_inches='tight')       



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
#lgb.plot_importance(model_lgb)
#pyplot.show()