
# coding: utf-8

# In[8]:


## Laden von der verschieden Bibliothek zur Daten Visualizierung und Vorhersagen
import os
os.chdir('/home/jupyter/jupyterNotebooks/m4competition18/data')
import numpy as np
# from __future__ import division
from matplotlib import pyplot
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.plotting import autocorrelation_plot
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression ,BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import rmsprop
from keras import backend as ker
from math import sqrt
import tensorflow as tf
import prediction
import pprint
from statsmodels.tsa.arima_model import ARIMA
#os.getcwd()


# Load of Hourly Data
df_hourly = pd.read_csv("Hourly-train.csv", skiprows=0, index_col =0)
Dataset_hourly = df_hourly.T

## Laden von Dataset Weekly-train
df_weekly = pd.read_csv("Weekly-train.csv", skiprows=0, index_col =0)
Dataset_weekly = df_weekly.T

## Laden von  Dataset Yearly-train
df_yearly = pd.read_csv("Yearly-train.csv", skiprows=0, index_col =0)
Dataset_yearly = df_yearly.T

# Load of monthly Data
df_monthly = pd.read_csv("Monthly-train.csv", skiprows=0, index_col =0)
Dataset_monthly = df_monthly.T

## Laden von Dataset Quarterly-train
df_quaterly = pd.read_csv("Quarterly-train.csv", skiprows=0, index_col =0)
Dataset_quaterly = df_quaterly.T

# load of Daily Data
df_daily = pd.read_csv("Daily-train.csv", skiprows=0, index_col =0)
Dataset_daily = df_daily.T




### Die Modellen



def Lin_Reg():
    model = LinearRegression(normalize=True)
    return model

def Ridge_Regression():
    model =  BayesianRidge(compute_score=True)
    return model   

def Dtree_Regression():
    model = DecisionTreeRegressor(criterion='mae',max_depth=28,min_samples_split=5,
                                 min_samples_leaf =5)
    return model

def K_neaRegression ():
    model = KNeighborsRegressor(n_neighbors=5)
    return model


def SVM_Regression ():
    model =  SVR(kernel='rbf', C=1e3, gamma=0.1)
    return model 



def mlp_benchm():            ## BENCHMARK ##
    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    return model



# Reihe der Modelle
### Linear Regression (Lin_Reg)                      ---> i=0
### Decision Tree (Dtree_Regression)                 ---> i=1
### K nearest Neighbours (K_neaRegression)           ---> i=2 
### K nearest Neighbours (SVM_Regression)            ---> i=3 
### Multi-Layers Perceptron [BENCHMARK] (mlp_benchm) ---> i=4 
### simple recurent nn [BENCHMARK] (rnn_benchm)      ---> i=5 
### Lstm neuronal network (LSTM)                     ---> i=6 



def main_prediction(i,model):
    T_sMape = []
    T_Mase = []
            
    # Hourly Daten
    print('Dataset_hourly')
    
    sMape_hourly_general,Mase_hourly_general = prediction.Prediction().main_resultat (Dataset_hourly,model,48,24,i)
    T_sMape.append(sMape_hourly_general)
    T_Mase.append(Mase_hourly_general)
     # Daily daten    
    print('###  Daily daten  ###')
    
    sMape_daily_general, Mase_daily_general = prediction.Prediction().main_resultat (Dataset_daily,model,14,1,i)
    T_sMape.append(sMape_daily_general)
    T_Mase.append(Mase_daily_general) 
            
    # Weekly daten
    print('###  Weekly daten  ###')
    sMape_weekly_general,Mase_weekly_general = prediction.Prediction().main_resultat (Dataset_weekly,model,13,1,i)
    T_sMape.append(sMape_weekly_general)
    T_Mase.append(Mase_weekly_general)   
    # Monthly Daten
    print('###  Monthly daten  ###')
    sMape_monthly_general,Mase_monthly_general= prediction.Prediction().main_resultat (Dataset_monthly,model,18,12,i)
    T_sMape.append(sMape_monthly_general)
    T_Mase.append(Mase_monthly_general) 
    # Quaterly Daten
    print('###  Quaterly daten  ###')
    sMape_quaterly_general,Mase_quaterly_general= prediction.Prediction().main_resultat (Dataset_quaterly,model,8,4,i)
    T_sMape.append(sMape_quaterly_general)
    T_Mase.append(Mase_quaterly_general)
    # Yearly Daten
    print('###  Yearly daten  ###')
    sMape_yearly_general,Mase_yearly_general =prediction.Prediction().main_resultat (Dataset_yearly,model,6,1,i)
    T_sMape.append(sMape_yearly_general)
    T_Mase.append(Mase_yearly_general)
    OWA = [np.mean(T_sMape),np.mean(T_Mase)]
  
    p = [model,sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
         sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
         Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
         Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase),np.mean(OWA)
                  ]
    return p

def main_prediction_one_data(i,model,data,fh,freq):
   
    sMape_general,Mase_general = prediction.Prediction().main_resultat (data,model,fh,freq,i)
    p = [model,sMape_general,Mase_general]
    print( "smape: ",sMape_general)
    print(" mase: ",Mase_general)
    
    return p

         
        
def global_prediction_one_model(i,model):    
    columnsname= ["Model","sMape Hourly","sMape Daily","sMape Weekly","sMape Monthly","sMape Quaterly","sMape Yearly","sMAPE TOTAL",
                 "Mase Hourly","Mase Daily","Mase Weekly","Mase Monthly","Mase Quaterly","Mase Yearly","MAPE TOTAL", "OWA"]
    ds = pd.DataFrame(columns=columnsname )
    ds.to_csv('output_one.csv')
    p = main_prediction(i,model)
    ds.loc[i] = p        
    ds=ds.round(4)
    ds.to_csv('output_one.csv', mode='a', header=False)
    print(p)

def global_prediction():
    m = []
    m.append(Lin_Reg())
    m.append(Dtree_Regression())
    m.append(K_neaRegression())
    m.append(SVM_Regression())
    m.append(mlp_benchm())
    m.append("Simple RNN BENCHMARK")
    m.append("LSTM Neural network")
    columnsname= ["Model","sMape Hourly","sMape Daily","sMape Weekly","sMape Monthly","sMape Quaterly","sMape Yearly","sMAPE TOTAL",
                 "Mase Hourly","Mase Daily","Mase Weekly","Mase Monthly","Mase Quaterly","Mase Yearly","MAPE TOTAL", "OWA"]
    ds = pd.DataFrame(columns=columnsname )
    ds.to_csv('out.csv')
    

    for i in range (len(m)):
        print("***MODEL:", i,"***\n")
        p = main_prediction(i,m[i])
        print(p)
        ds.loc[i] = p        
        ds=ds.round(4)
        ds.to_csv('out.csv', mode='a', header=False)
    print("done")
        

       


# In[9]:


global_prediction()


# In[6]:




