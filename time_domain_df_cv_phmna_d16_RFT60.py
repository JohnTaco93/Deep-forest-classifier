#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings
warnings.filterwarnings("ignore")
print('###################### time domain started d16 RFT 60 #########################')
import pandas as pd
import numpy as np
import random
import gc
import numpy as np
import scipy.io
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# In[2]:


def prepare_samples(data_pdmp,data_pin,data_po):
    
    data_pdmp=data_pdmp.fillna(0)
    data_pin=data_pin.fillna(0)
    data_po=data_po.fillna(0)
    
    X=[]
    for i in range(data_pdmp.shape[0]):
        pdmp_signal=data_pdmp.iloc[i,1:].values
        pin_signal=data_pin.iloc[i,1:].values
        po_signal=data_po.iloc[i,1:].values
        X_i=np.array([pdmp_signal,pin_signal,po_signal])
        X.append(X_i)
    
    return np.array(X)   


# In[3]:
path_data='C:/Users/JohnTaco/Desktop/john/tesis/phmna/Data_Challenge_PHM2022_training_data_mod/'

data_pdmp1=pd.read_csv(path_data+'data_pdmp1.csv',header=None)
data_pin1=pd.read_csv(path_data+'data_pin1.csv',header=None)
data_po1=pd.read_csv(path_data+'data_po1.csv',header=None)

data_pdmp2=pd.read_csv(path_data+'data_pdmp2.csv',header=None)
data_pin2=pd.read_csv(path_data+'data_pin2.csv',header=None)
data_po2=pd.read_csv(path_data+'data_po2.csv',header=None)

data_pdmp3=pd.read_csv(path_data+'data_pdmp3.csv',header=None)
data_pin3=pd.read_csv(path_data+'data_pin3.csv',header=None)
data_po3=pd.read_csv(path_data+'data_po3.csv',header=None)

data_pdmp4=pd.read_csv(path_data+'data_pdmp4.csv',header=None)
data_pin4=pd.read_csv(path_data+'data_pin4.csv',header=None)
data_po4=pd.read_csv(path_data+'data_po4.csv',header=None)

data_pdmp5=pd.read_csv(path_data+'data_pdmp5.csv',header=None)
data_pin5=pd.read_csv(path_data+'data_pin5.csv',header=None)
data_po5=pd.read_csv(path_data+'data_po5.csv',header=None)

data_pdmp6=pd.read_csv(path_data+'data_pdmp6.csv',header=None)
data_pin6=pd.read_csv(path_data+'data_pin6.csv',header=None)
data_po6=pd.read_csv(path_data+'data_po6.csv',header=None)



# In[4]:

#%%time
X_m1=prepare_samples(data_pdmp1,data_pin1,data_po1)
X_m2=prepare_samples(data_pdmp2,data_pin2,data_po2)
X_m3=prepare_samples(data_pdmp3,data_pin3,data_po3)
X_m4=prepare_samples(data_pdmp4,data_pin4,data_po4)
X_m5=prepare_samples(data_pdmp5,data_pin5,data_po5)
X_m6=prepare_samples(data_pdmp6,data_pin6,data_po6)

# In[5]:

y_1=data_pdmp1[0]
y_2=data_pdmp2[0]
y_3=pd.read_csv(path_data+'machine3_100_lb.txt',header=None)[0].values
y_4=data_pdmp4[0]
y_5=data_pdmp5[0]
y_6=data_pdmp6[0]

# In[6]:


y_1=y_1-1
y_2=y_2-1
y_3=y_3-1
y_4=y_4-1
y_5=y_5-1
y_6=y_6-1


# In[7]:


print(X_m1.shape,X_m2.shape,X_m4.shape,X_m5.shape,X_m6.shape)


# In[8]:


X=[X_m1,X_m2,X_m3,X_m4,X_m5,X_m6]
y=[y_1,y_2,y_3,y_4,y_5,y_6]


# In[9]:


from GCForest_multi_sec import gcForest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[10]:


import time
start_time = time.time()
acc_train_all=[]
acc_val_all=[]
for j in range(6):
    print(j)
    machine_numbers=[0,1,2,3,4,5]
    x_val=X[j][:,:,0:692]
    y_val=y[j]
    machine_numbers.remove(j)
    x_tr=[X[k][0:300,:,0:692] for k in machine_numbers]
    y_tr=[y[k][0:300] for k in machine_numbers]
    
    x_tr=np.concatenate(x_tr)
    y_tr=np.concatenate(y_tr)
    x_tr=np.array(x_tr)
    y_tr=np.array(y_tr)
    x_val=np.array(x_val)
    y_val=np.array(y_val)
    
    print(x_tr.shape,x_val.shape,y_tr.shape,y_val.shape)
    shape_1X=[3, 692]
    gcf = gcForest(shape_1X=[3, 692], window=(shape_1X[1]//16), min_samples_mgs=10, min_samples_cascade=7, n_mgsRFtree=60, n_jobs=-1)
    X_tr_mgs = gcf.mg_scanning(x_tr, y_tr)
    X_vl_mgs = gcf.mg_scanning(x_val)
    #X_te_mgs = gcf.mg_scanning(X_test)
    gcf = gcForest(tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_jobs=-1)
    _ = gcf.cascade_forest(X_tr_mgs, y_tr)
    
     #TRAINING
    pred_proba = gcf.cascade_forest(X_tr_mgs)
    tmp = np.mean(pred_proba, axis=0)
    y_hat_train = np.argmax(tmp, axis=1)
    acc_train=accuracy_score(y_true=y_tr, y_pred=y_hat_train)
    #f1_train=f1_score(y_true=y_tr, y_pred=y_hat_train)
    #recall_train=recall_score(y_true=y_tr, y_pred=y_hat_train)
    #precision_train=precision_score(y_true=y_tr, y_pred=y_hat_train)
    
    acc_train_all.append(acc_train)
    #f1_train_all.append(f1_train)
    #recall_train_all.append(recall_train)
    #precision_train_all.append(precision_train)

    #VALIDATION
    pred_proba = gcf.cascade_forest(X_vl_mgs)
    tmp = np.mean(pred_proba, axis=0)
    y_hat_val = np.argmax(tmp, axis=1)
    acc_val=accuracy_score(y_true=y_val, y_pred=y_hat_val)
    #f1_val=f1_score(y_true=y_val, y_pred=y_hat_val)
    #recall_val=recall_score(y_true=y_val, y_pred=y_hat_val)
    #precision_val=precision_score(y_true=y_val, y_pred=y_hat_val)
    
    print(acc_val)
    
    acc_val_all.append(acc_val)
    #f1_val_all.append(f1_val)
    #recall_val_all.append(recall_val)
    #precision_val_all.append(precision_val)


# In[11]:
print('results time domain started d16 RFT 60')
print(acc_val_all)
print('accuracy train',np.mean(acc_train_all))
print('accuracy test',np.mean(acc_val_all))
print("--- %s seconds ---" % (time.time() - start_time))



