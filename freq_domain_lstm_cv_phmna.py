#!/usr/bin/env python
# coding: utf-8

# In[1]:

print('############################################ freq domain started lstm...#################################')
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
from scipy.fft import fft, ifft


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
        
        pdmp_signal_freq=np.abs(fft(pdmp_signal))*2/len(pdmp_signal)
        pdmp_signal_freq=pdmp_signal_freq[0:len(pdmp_signal_freq)//2]
        
        pin_signal_freq=np.abs(fft(pin_signal))*2/len(pin_signal)
        pin_signal_freq=pin_signal_freq[0:len(pin_signal_freq)//2]
        
        po_signal_freq=np.abs(fft(po_signal))*2/len(po_signal)
        po_signal_freq=po_signal_freq[0:len(po_signal_freq)//2]

        X_i=np.array([pdmp_signal_freq,pin_signal_freq,po_signal_freq])
        X.append(X_i)
    
    return np.array(X)   


# In[3]:


data_pdmp1=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp1.csv',header=None)
data_pin1=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin1.csv',header=None)
data_po1=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po1.csv',header=None)

data_pdmp2=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp2.csv',header=None)
data_pin2=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin2.csv',header=None)
data_po2=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po2.csv',header=None)

data_pdmp3=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp3.csv',header=None)
data_pin3=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin3.csv',header=None)
data_po3=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po3.csv',header=None)

data_pdmp4=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp4.csv',header=None)
data_pin4=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin4.csv',header=None)
data_po4=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po4.csv',header=None)

data_pdmp5=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp5.csv',header=None)
data_pin5=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin5.csv',header=None)
data_po5=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po5.csv',header=None)

data_pdmp6=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pdmp6.csv',header=None)
data_pin6=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_pin6.csv',header=None)
data_po6=pd.read_csv('Data_Challenge_PHM2022_training_data_mod/data_po6.csv',header=None)


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
y_3=pd.read_csv('machine3_100_lb.txt',header=None)[0].values
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


min_len=np.min((X_m1.shape[2],X_m2.shape[2],X_m3.shape[2],X_m4.shape[2],X_m5.shape[2],X_m6.shape[2]))


# In[9]:


X=[X_m1,X_m2,X_m3,X_m4,X_m5,X_m6]
y=[y_1,y_2,y_3,y_4,y_5,y_6]


# In[10]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.io
from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers import Activation

# In[11]:

def build_model_lstm(input_shape):  # random search passes this hyperparameter() object 

    
    optm=tf.keras.optimizers.Adam(learning_rate=0.001)

    model = Sequential()
    model.add(LSTM(256,input_shape=input_shape,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(LSTM(units=128,input_shape=input_shape,return_sequences=True))  
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(LSTM(units=128,input_shape=input_shape,return_sequences=False))  
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(11, activation='softmax'))

    model.compile(optimizer=optm,
                  loss='categorical_crossentropy',
    #              loss='sigmoid_cross_entropy_with_logits',
                  metrics=["accuracy"])
    
    return model

# In[12]:
import time
start_time = time.time()

#%%time
acc_train_all=[]
acc_val_all=[]
for j in range(6):
    print(j)
    machine_numbers=[0,1,2,3,4,5]
    x_val=X[j][:,:,0:min_len]
    y_val=y[j]
    machine_numbers.remove(j)
    x_tr=[X[k][0:300,:,0:min_len] for k in machine_numbers]
    y_tr=[y[k][0:300] for k in machine_numbers]

    x_tr=np.concatenate(x_tr)
    y_tr=np.concatenate(y_tr)
    x_tr=np.array(x_tr)
    y_tr=np.array(y_tr)
    x_val=np.array(x_val)
    y_val=np.array(y_val)

    #x_tr=[x_tr[k].T for k in range(x_tr.shape[0])]
    x_tr=np.array(x_tr)
    #x_val=[x_val[k].T for k in range(x_val.shape[0])]
    x_val=np.array(x_val)

    y_tr=pd.get_dummies(y_tr).values
    y_val=pd.get_dummies(y_val).values
    
    print(x_tr.shape,x_val.shape,y_tr.shape,y_val.shape)

    # training
    model=build_model_lstm(x_tr[0].shape)
    model.fit(x_tr,y_tr,epochs=500, verbose=0)

    #predictions
    #training data
    y_hat_train=model.predict(x_tr)
    y_hat_train=[(y_hat_train[j] == y_hat_train[j].max()).view(np.int8) for j in range(y_hat_train.shape[0])]
    y_hat_train=np.array(y_hat_train)
    acc_train=accuracy_score(y_true=y_tr, y_pred=y_hat_train)
    print(acc_train)
    acc_train_all.append(acc_train)

    #test data
    y_hat_val=model.predict(x_val)
    y_hat_val=[(y_hat_val[j] == y_hat_val[j].max()).view(np.int8) for j in range(y_hat_val.shape[0])]
    y_hat_val=np.array(y_hat_val)
    acc_val=accuracy_score(y_true=y_val, y_pred=y_hat_val)
    print(acc_val)
    acc_val_all.append(acc_val)


# In[13]:


print(acc_val_all)
print('results freq domain started lstm')
print('accuracy train',np.mean(acc_train_all))
print('accuracy test',np.mean(acc_val_all))
print("--- %s seconds ---" % (time.time() - start_time))




