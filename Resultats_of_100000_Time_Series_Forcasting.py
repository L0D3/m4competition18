
# coding: utf-8

# In[7]:


## Laden von der verschieden Bibliothek zur Daten Visualizierung und Vorhersagen
import os
import numpy as np
import zipfile
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
from sklearn.ensemble import GradientBoostingRegressor
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
os.chdir('/home/jupyter/jupyterNotebooks/m4competition18/data')
os.getcwd()
from pprint import pprint

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

def remov_nan (dataset):
    '''
    to remove all NaN Values in a 
    Time Serie Dataframe
    '''
    n = dataset.isnull().sum() 
    data = dataset[0:(len(dataset)-n)]
    return data

def copy_val(x):
    '''
    to copy a list or array in a new memory 
    without reference 
    x: list or array
    '''
    y =[]
    for i in x:
        y.append(i)
    return np.array(y)

def normalisieren_data(dataset):
    '''
    to normalize Data
    : dataset : Data to normalize
    ''' 
    scaler = scaler =MinMaxScaler(feature_range=(0, 1)).fit(dataset)
    Dataset_normalized = scaler.transform(dataset)
    return Dataset_normalized,scaler

def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


## BENCHMARK ##
def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    if len(ts_init) % 2 == 0:
        ts_ma = pd.rolling_mean(ts_init, window, center=True)
        ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
        ts_ma = np.roll(ts_ma, -1)
    else:
        ts_ma = pd.rolling_mean(ts_init, window, center=True)

    return ts_ma

def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)

## BENCHMARK ##
def smape(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item() 

##===Mean Absolute Scaled Error ====##
def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])
    masep = np.mean(abs(insample[freq:] - y_hat_naive))
    return np.mean(abs(y_test - y_hat_test)) / masep
#Hier wird die "Time serie" als "spervised learning Problem" umgewandel.
#Die Datenmenge der Zeitreihen wird in Training und Testing Datamenge und jeweils in input & output Daten

# Hilfsfunktion , die eine Datenmenge in input und output Menge aufteile 
def split_input_output(dataset: np.ndarray, in_back: int=1) -> (np.ndarray, np.ndarray):
    """ 
    The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
    and the `in_back`, which is the number of previous time steps to use as input variables
    to predict the next time period — in this case defaulted to 1.
    :dataset: numpy dataset
    :in_variable: number of previous time steps as int
    :return: tuple of input and output dataset
    """
    Input, Output = [], []
    for i in range(len(dataset)-in_back):
        a = dataset[i:(i+in_back)]
        Input.append(a)
        Output.append(dataset[i + in_back])
    return np.array(Input), np.array(Output)

## Folgende Funktion split die Datenmende in Training and Testing Daten.

def split_into_train_test(dataset: np.ndarray,train_size, in_back) -> (np.ndarray, np.ndarray):
    """
    Splits dataset into training and test datasets. 
    : dataset: (np.ndarray) Time serie Dataset 
    : train_size: (int) Größe der Training Datamenge
    : look_back: (int) number of previous time steps 
    :return: tuple of training data and test dataset
    """
    if not train_size > in_back:
        raise ValueError('train_size muss größer als look_back',"train_size:",train_size,"in_back:",in_back)
    train= dataset[0:train_size]
    test = dataset[train_size - in_back:len(dataset)]
    #print('train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
    return train, test

## Es wird hier die Datenmenge in X_train,Y_train für das Training und X_test,Y_test für das Testing 

def all_split (dataset: np.ndarray,fh, in_back) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Splits dataset into input-training (X_train), outout_training(Y_train) and input_test(X_test) , output_test(Y_test) datasets.
    : dataset:(np.ndarray) Time serie Dataset
    :df:(float64) Größe der Testing Datamenge 
    : in_back: (int) number of previous time steps 
    :return: x_train, y_train, x_test, y_test
    """
    #if not (size_prozent>0 and size_prozent<1):
        #raise ValueError('size_prozent of training must be in the interval 0 and 1')
    train_size = len(dataset)-fh
    training, testing = split_into_train_test(dataset,train_size,in_back)
    X_train, Y_train = split_input_output(training,in_back)
    X_test, Y_test = split_input_output(testing,in_back)
    return X_train,Y_train,X_test[0].reshape(1,-1),Y_test

def check_pred (dataset: pd.DataFrame,y_pred: np.ndarray):
    ''''
    this function check the negativity of the predicted values, set them to null 
    if they are negativ and to max value of the serie data if they are extrem high
    : dataset: Dataset of the serie
    : y_pred:  The list of predicted values
    : return:
    '''
    for i in range(len(y_pred)):
        if y_pred[i]<0:
            y_pred[i]=0
        if y_pred[i]> (9000*max(dataset)):
            y_pred[i]=max(dataset)

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

## BENCHMARK ##

def mlp_benchm():
    model = MLPRegressor(hidden_layer_sizes=6, activation='identity', solver='adam',
                         max_iter=100, learning_rate='adaptive', learning_rate_init=0.001,
                         random_state=42)
    return model


def rnn_bench(x_train, y_train, x_test, fh, input_size):
    """
    Forecasts using 6 SimpleRNN nodes in the hidden layer and a Dense output layer
    :param x_train: train data
    :param y_train: target values for training
    :param x_test: test data
    :param fh: forecasting horizon
    :param input_size: number of points used as input
    :return:
    """
    # reshape to match expected input
    x_train = np.reshape(x_train, (-1, input_size, 1))
    x_test = np.reshape(x_test, (-1, input_size, 1))
    # create the model
    model = Sequential([
        SimpleRNN(6, input_shape=(input_size, 1), activation='linear',
                  use_bias=False, kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal', bias_initializer='zeros',
                  dropout=0.0, recurrent_dropout=0.0),
        Dense(1, use_bias=True, activation='linear')
    ])
    opt = rmsprop(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)
    # fit the model to the training data
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
    # make predictions
    y_hat_test = []
    last_prediction = model.predict(x_test)[0][0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0][(len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0][0]
        #print(last_prediction)
    return np.asarray(y_hat_test)

def LSTM_NN(x_train, y_train, x_test, fh, in_back):
    
    # reshape to match expected input
    x_train = np.reshape(x_train, (x_train.shape[0], in_back, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], in_back, 1))
    # create the model
    model = Sequential()
    model.add(LSTM(10, input_shape=(in_back,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)
    # make predictions
    y_hat_test = []
    last_prediction = model.predict(x_test)[0][0]
    for i in range(0, fh):
        y_hat_test.append(last_prediction)
        x_test[0] = np.roll(x_test[0], -1)
        x_test[0][(len(x_test[0]) - 1)] = last_prediction
        last_prediction = model.predict(x_test)[0][0]
        #print(last_prediction)
    return np.asarray(y_hat_test)


## Eine Funktion, die die Vorhersage von mehreren Time-series mit simple Reccurent neuronal netzwerk(Benchmark model) evaluiert. 
## Sie gibt die durchschnittliche Werte der Metriken sMAPE und MASE für die Menge der Time-Series zurück.
def main_predict_rnn (Data: pd.DataFrame,fh,freq):
   
    
    n = Data.shape[1]   # number of Serie in the Dataset
    
    model_MASE =[]     # a list to save all mase_values of linear regression of each Serie forcasting 
    model_sMAPE =[]    # a list to save all smape_values of linear regression of each Serie forcasting
    
    
    # Iteration through each serie in the dataset
    
    for i in range(n):
        
        Dat = Data.iloc[:,i]
        
        # remove all NaN value from the serie
        Data_val  = remov_nan (Dat)
       
        in_back = int(0.2*len(Data_val ))
        # load the value of new_Data
        # = new_Data.values
        
        # ==== remove seasonality ====#
        seasonality_in = deseasonalize(Data_val, freq)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * 100 / seasonality_in[i % freq]

        # ==== detrending ====#
        a, b = detrend(Data_val)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] - ((a * i) + b)
        
        
        # Split the data into x_train, y_train, x_test, y_test
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        
        # Prediction with Linear regression
        Y_pred = rnn_bench(x_train, y_train, copy_val(x_test),fh, in_back)
      
     # ==== add trend ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] + ((a * i) + b)
    
        for i in range(0, fh):
            Y_pred[i] = Y_pred[i] + ((a * (len(Data_val) + i + 1)) + b)
           
    # ==== add seasonality ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * seasonality_in[i % freq] / 100
    
        for i in range(len(Data_val), len(Data_val) + fh):
            Y_pred[i - len(Data_val)] = Y_pred[i - len(Data_val)] * seasonality_in[i % freq] / 100
        
        #print("y_pre",Y_pred)
        #print(Data_val.head)
    
        # check the prediction on negativity and extremity
        check_pred(Data_val,Y_pred)
        
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        # calculation of Error
        model_sMAPE.append(smape(y_test, Y_pred))
        model_MASE.append(mase(Data_val[:-fh], y_test, Y_pred, freq))
   
    return np.mean(model_sMAPE), np.mean(model_MASE)

def main_predict_lstm (Data: pd.DataFrame,fh,freq):
 
    n = Data.shape[1]   # number of Serie in the Dataset
    
    model_MASE =[]     # a list to save all mase_values of linear regression of each Serie forcasting 
    model_sMAPE =[]    # a list to save all smape_values of linear regression of each Serie forcasting
    
    
    # Iteration through each serie in the dataset
    
    for i in range(n):
        
        Dat = Data.iloc[:,i]
        
        # remove all NaN value from the serie
        Data_val  = remov_nan (Dat)
       
        in_back = int(0.2*len(Data_val ))
        # load the value of new_Data
        # = new_Data.values
        
        # ==== remove seasonality ====#
        seasonality_in = deseasonalize(Data_val, freq)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * 100 / seasonality_in[i % freq]

        # ==== detrending ====#
        a, b = detrend(Data_val)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] - ((a * i) + b)
        
        
        # Split the data into x_train, y_train, x_test, y_test
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        
        # Prediction with Linear regression
        Y_pred = LSTM_NN(x_train, y_train, copy_val(x_test),fh, in_back)
      
     # ==== add trend ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] + ((a * i) + b)
    
        for i in range(0, fh):
            Y_pred[i] = Y_pred[i] + ((a * (len(Data_val) + i + 1)) + b)
           
    # ==== add seasonality ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * seasonality_in[i % freq] / 100
    
        for i in range(len(Data_val), len(Data_val) + fh):
            Y_pred[i - len(Data_val)] = Y_pred[i - len(Data_val)] * seasonality_in[i % freq] / 100
        
        #print("y_pre",Y_pred)
        #print(Data_val.head)
    
        # check the prediction on negativity and extremity
        check_pred(Data_val,Y_pred)
        
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        # calculation of Error
        model_sMAPE.append(smape(y_test, Y_pred))
        model_MASE.append(mase(Data_val[:-fh], y_test, Y_pred, freq))
    return np.mean(model_sMAPE), np.mean(model_MASE)

def main_prediction_mlp (Data: pd.DataFrame,model,fh,freq): 
    n = Data.shape[1]   # number of Serie in the Dataset  
    Lin_Reg_MASE =[]     # a list to save all mase_values of linear regression of each Serie forcasting 
    Lin_Reg_sMAPE =[]    # a list to save all smape_values of linear regression of each Serie forcasting
    # Iteration through each serie in the dataset
    for i in range(n):
        zr = Data.iloc[:,i]  
        # remove all NaN value from the serie
        new_Data = remov_nan (zr)        
        in_back = int(0.2*len(new_Data))
        # load the value of new_Data
        Data_val = new_Data.values        
        # ==== remove seasonality ====#
        seasonality_in = deseasonalize(Data_val, freq)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * 100 / seasonality_in[i % freq]
        # ==== detrending ====#
        a, b = detrend(Data_val)
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] - ((a * i) + b)       
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        
        model.fit(x_train, y_train)
        
        predict =[]
        prediction_current = model.predict(x_test)[0]
      # Techniques of Iteration for the horizon forcasting
        for i in range(0, fh):
            # add the first prediction to y_predict
            predict.append(prediction_current)
            # move the first element in x_test to the last position, in order to remove 
            x_test[0] = np.roll(x_test[0], -1)
            # set now the current_prediction value at the last position of x_test
            x_test[0][(len(x_test[0]) - 1)] = prediction_current
            prediction_current = model.predict(x_test)[0]
        Y_pred = np.asarray(predict)     
     # ==== add trend ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] + ((a * i) + b)
    
        for i in range(0, fh):
            Y_pred[i] = Y_pred[i] + ((a * (len(Data_val) + i + 1)) + b)
           
    # ==== add seasonality ====#
        for i in range(0, len(Data_val)):
            Data_val[i] = Data_val[i] * seasonality_in[i % freq] / 100
    
        for i in range(len(Data_val), len(Data_val) + fh):
            Y_pred[i - len(Data_val)] = Y_pred[i - len(Data_val)] * seasonality_in[i % freq] / 100
        
        
        # check the prediction on negativity and extremity
        check_pred(new_Data,Y_pred)
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        # calculation of Error
        Lin_Reg_sMAPE.append(smape(y_test, Y_pred))
        Lin_Reg_MASE.append(mase(new_Data[:-fh], y_test, Y_pred, freq))
   
    return  np.mean(Lin_Reg_sMAPE),  np.mean(Lin_Reg_MASE)

def main_prediction_SVM (Data: pd.DataFrame,model,fh,freq):
 
    n = Data.shape[1]   # number of Serie in the Dataset 
    Lin_Reg_MASE =[]     # a list to save all mase_values of linear regression of each Serie forcasting 
    Lin_Reg_sMAPE =[]    # a list to save all smape_values of linear regression of each Serie forcasting
    # Iteration through each serie in the dataset
    for i in range(n):
        zr = Data.iloc[:,i]
        
        # remove all NaN value from the serie
        new_Data = remov_nan (zr)
        
        in_back = int(0.3*len(new_Data))
        # load the value of new_Data
        Data_val = new_Data.values
        
        rr ,s = normalisieren_data(Data_val.reshape(-1, 1))
        Data_val_norm = np.reshape(rr,len(Data_val))
        
        
        x_train,y_train,x_test,y_test = all_split(Data_val_norm,fh, in_back)
        
        model.fit(x_train,y_train)
        
        predict =[]
        prediction_current = model.predict(x_test)[0]
        # Techniques of Iteration for the horizon forcasting
        for i in range(0, fh):
            # add the first prediction to y_predict
            predict.append(prediction_current)
            # move the first element in x_test to the last position, in order to remove 
            x_test[0] = np.roll(x_test[0], -1)
            # set now the current_prediction value at the last position of x_test
            x_test[0][(len(x_test[0]) - 1)] = prediction_current
            prediction_current = model.predict(x_test)[0]
        Y_pred_LinReg = np.asarray(predict) 
        Y_pred = s.inverse_transform([Y_pred_LinReg])[0]
        
        # check the prediction on negativity and extremity
        check_pred(new_Data,Y_pred)
        
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        # calculation of Error
        Lin_Reg_sMAPE.append(smape(y_test, Y_pred))
        Lin_Reg_MASE.append(mase(new_Data[:-fh], y_test, Y_pred, freq))
    return  np.mean(Lin_Reg_sMAPE),  np.mean(Lin_Reg_MASE)

def main_prediction(Data: pd.DataFrame,model,fh,freq):   
    n = Data.shape[1]   # number of Serie in the Dataset  
    Lin_Reg_MASE =[]     # a list to save all mase_values of linear regression of each Serie forcasting 
    Lin_Reg_sMAPE =[]    # a list to save all smape_values of linear regression of each Serie forcasting

    # Iteration through each serie in the dataset
    for i in range(n):
        zr = Data.iloc[:,i]
        
        # remove all NaN value from the serie
        new_Data = remov_nan (zr)
        
        in_back = int(0.2*len(new_Data))
        # load the value of new_Data
        Data_val = new_Data.values
        
        x_train,y_train,x_test,y_test = all_split(Data_val,fh, in_back)
        
        model.fit(x_train,y_train)
        
        predict =[]
        prediction_current = model.predict(x_test)[0]
        # Techniques of Iteration for the horizon forcasting
        for i in range(0, fh):
            # add the first prediction to y_predict
            predict.append(prediction_current)
            # move the first element in x_test to the last position, in order to remove 
            x_test[0] = np.roll(x_test[0], -1)
            # set now the current_prediction value at the last position of x_test
            x_test[0][(len(x_test[0]) - 1)] = prediction_current
            prediction_current = model.predict(x_test)[0]
        Y_pred_LinReg = np.asarray(predict)  
        # check the prediction on negativity and extremity
        check_pred(new_Data,Y_pred_LinReg)       
        # calculation of Error
        Lin_Reg_sMAPE.append(smape(y_test, Y_pred_LinReg))
        Lin_Reg_MASE.append(mase(new_Data[:-fh], y_test, Y_pred_LinReg, freq))
    return  np.mean(Lin_Reg_sMAPE),  np.mean(Lin_Reg_MASE)



def global_prediction():
    
    c = 0
    m = []
    m.append(Lin_Reg())
    m.append(Dtree_Regression())
    m.append(K_neaRegression())
    #m.append(Ridge_Regression())
    m.append(SVM_Regression())
    m.append(mlp_benchm())
    m.append("rnn_bench")
    m.append("LSTM_NN")
    b = np.array([])
    columnsname= ["Model","sMape Hourly","sMape Daily","sMape Weekly","sMape Monthly","sMape Quaterly","sMape Yearly","sMAPE TOTAL",
                 "Mase Hourly","Mase Daily","Mase Weekly","Mase Monthly","Mase Quaterly","Mase Yearly","MAPE TOTAL", ]
    ds = pd.DataFrame(columns=columnsname )
    ds.to_csv('out.csv')

    for i in range (len(m)):
        print(i)
        if i<3:
            T_sMape = []
            T_Mase = []
    # Hourly Daten
            print('Dataset_hourly')

            sMape_hourly_general,Mase_hourly_general = main_prediction(Dataset_hourly,m[i],48,24)
            T_sMape.append(sMape_hourly_general)
            T_Mase.append(Mase_hourly_general)
     # Daily daten    
            print('Daily daten')
            sMape_daily_general, Mase_daily_general = main_prediction(Dataset_daily,m[i],14,1)
            T_sMape.append(sMape_daily_general)
            T_Mase.append(Mase_daily_general) 
            
    # Weekly daten
            print('Weekly daten')
            sMape_weekly_general,Mase_weekly_general = main_prediction(Dataset_weekly,m[i],13,1)
            T_sMape.append(sMape_weekly_general)
            T_Mase.append(Mase_weekly_general)   
    # Monthly Daten
            print('Monthly daten')
            sMape_monthly_general,Mase_monthly_general= main_prediction(Dataset_monthly,m[i],18,12)
            T_sMape.append(sMape_monthly_general)
            T_Mase.append(Mase_monthly_general) 
    # Quaterly Daten
            sMape_quaterly_general,Mase_quaterly_general= main_prediction(Dataset_quaterly,m[i],8,4)
            T_sMape.append(sMape_quaterly_general)
            T_Mase.append(Mase_quaterly_general)
    # Yearly Daten
            print('Yearly daten')
            sMape_yearly_general,Mase_yearly_general =main_prediction(Dataset_yearly,m[i],6,1)
            T_sMape.append(sMape_yearly_general)
            T_Mase.append(Mase_yearly_general)
  
            p = [m[i],sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
                   sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
                   Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
                   Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase)
                  ]
            pprint(p)
            ds.loc[i] = p        
            ds=ds.round(4)
            ds.to_csv('out.csv', mode='a', header=False)

        ########################################################################################################
        
        
        if i == 3:                     ## Schleife von support vector Regression
            T_sMape = []
            T_Mase = [] 
     # Hourly Daten
            sMape_hourly_general,Mase_hourly_general = main_prediction_SVM(Dataset_hourly,m[i],48,24)
            T_sMape.append(sMape_hourly_general)
            T_Mase.append(Mase_hourly_general)
     # Daily daten  
            sMape_daily_general, Mase_daily_general = main_prediction_SVM(Dataset_daily,m[i],14,1)
            T_sMape.append(sMape_daily_general)
            T_Mase.append(Mase_daily_general)   
    # Weekly daten
            sMape_weekly_general,Mase_weekly_general = main_prediction_SVM(Dataset_weekly,m[i],13,1)
            T_sMape.append(sMape_weekly_general)
            T_Mase.append(Mase_weekly_general)  
    # Monthly Daten
            sMape_monthly_general,Mase_monthly_general= main_prediction_SVM(Dataset_monthly,m[i],18,12)
            T_sMape.append(sMape_monthly_general)
            T_Mase.append(Mase_monthly_general)    
    # Quaterly Daten
            sMape_quaterly_general,Mase_quaterly_general= main_prediction_SVM(Dataset_quaterly,m[i],8,4)
            T_sMape.append(sMape_quaterly_general)
            T_Mase.append(Mase_quaterly_general)
    # Yearly Daten
            sMape_yearly_general,Mase_yearly_general =main_prediction_SVM(Dataset_yearly,m[i],6,1)
            T_sMape.append(sMape_yearly_general)
            T_Mase.append(Mase_yearly_general)
  
 
            p = [m[i],sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
                   sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
                   Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
                   Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase)
                  ]
            pprint(p)
            ds.loc[i] = p        
            ds=ds.round(4)
            ds.to_csv('out.csv', mode='a', header=False)
      
    ###############################################################################################################
    
        if i == 4:                     ## Schleife von mlp Bench
            T_sMape = []
            T_Mase = [] 
     # Hourly Daten
            sMape_hourly_general,Mase_hourly_general = main_prediction_mlp(Dataset_hourly,m[i],48,24)
            T_sMape.append(sMape_hourly_general)
            T_Mase.append(Mase_hourly_general)
     # Daily daten  
            sMape_daily_general, Mase_daily_general = main_prediction_mlp(Dataset_daily,m[i],14,1)
            T_sMape.append(sMape_daily_general)
            T_Mase.append(Mase_daily_general)   
    # Weekly daten
            sMape_weekly_general,Mase_weekly_general = main_prediction_mlp(Dataset_weekly,m[i],13,1)
            T_sMape.append(sMape_weekly_general)
            T_Mase.append(Mase_weekly_general)  
    # Monthly Daten
            sMape_monthly_general,Mase_monthly_general= main_prediction_mlp(Dataset_monthly,m[i],18,12)
            T_sMape.append(sMape_monthly_general)
            T_Mase.append(Mase_monthly_general)    
    # Quaterly Daten
            sMape_quaterly_general,Mase_quaterly_general= main_prediction_mlp(Dataset_quaterly,m[i],8,4)
            T_sMape.append(sMape_quaterly_general)
            T_Mase.append(Mase_quaterly_general)
    # Yearly Daten
            sMape_yearly_general,Mase_yearly_general =main_prediction_mlp(Dataset_yearly,m[i],6,1)
            T_sMape.append(sMape_yearly_general)
            T_Mase.append(Mase_yearly_general)
  
            p = [m[i],sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
                   sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
                   Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
                   Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase)
                  ]
            pprint(p)
            ds.loc[i] = p        
            ds=ds.round(4)
            ds.to_csv('out.csv', mode='a', header=False)
      
        #print(.shape)
       
    ###########################################################################################################
    
        if i == 5:                     ## Schleife von RNN Bench
            T_sMape = []
            T_Mase = [] 
     # Hourly Daten
            sMape_hourly_general,Mase_hourly_general = main_predict_rnn(Dataset_hourly,48,24)
            T_sMape.append(sMape_hourly_general)
            T_Mase.append(Mase_hourly_general)
     # Daily daten
            sMape_daily_general, Mase_daily_general = main_predict_rnn(Dataset_daily,14,1)
            T_sMape.append(sMape_daily_general)
            T_Mase.append(Mase_daily_general)  
    # Weekly daten
            sMape_weekly_general,Mase_weekly_general = main_predict_rnn(Dataset_weekly.iloc[:,0:2],13,1)
            T_sMape.append(sMape_weekly_general)
            T_Mase.append(Mase_weekly_general)  
    # Monthly Daten
            sMape_monthly_general,Mase_monthly_general= main_predict_rnn(Dataset_monthly,18,12)
            T_sMape.append(sMape_monthly_general)
            T_Mase.append(Mase_monthly_general)
    # Quaterly Daten
            sMape_quaterly_general,Mase_quaterly_general= main_predict_rnn(Dataset_quaterly,8,4)
            T_sMape.append(sMape_quaterly_general)
            T_Mase.append(Mase_quaterly_general)
    # Yearly Daten
            sMape_yearly_general,Mase_yearly_general =main_predict_rnn(Dataset_yearly,6,1)
            T_sMape.append(sMape_yearly_general)
            T_Mase.append(Mase_yearly_general)
  
            p = [m[i],sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
                   sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
                   Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
                   Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase)
                  ]
            pprint(p)
            ds.loc[i] = p        
            ds=ds.round(4)
            ds.to_csv('out.csv', mode='a', header=False)
      
        
        ########################################################################################################
        
        if i == 6:                ## Schleife von LSTM_NN
            T_sMape = []
            T_Mase = [] 
     # Hourly Daten
            sMape_hourly_general,Mase_hourly_general = main_predict_lstm(Dataset_hourly,48,24)
            T_sMape.append(sMape_hourly_general)
            T_Mase.append(Mase_hourly_general)
     # Daily daten    
            sMape_daily_general, Mase_daily_general = main_predict_lstm(Dataset_daily,14,1)
            T_sMape.append(sMape_daily_general)
            T_Mase.append(Mase_daily_general)   
    # Weekly daten
            sMape_weekly_general,Mase_weekly_general = main_predict_lstm(Dataset_weekly,13,1)
            T_sMape.append(sMape_weekly_general)
            T_Mase.append(Mase_weekly_general)    
    # Monthly Daten
            sMape_monthly_general,Mase_monthly_general= main_predict_lstm(Dataset_monthly,18,12)
            T_sMape.append(sMape_monthly_general)
            T_Mase.append(Mase_monthly_general)  
    # Quaterly Daten
            sMape_quaterly_general,Mase_quaterly_general= main_predict_lstm(Dataset_quaterly,8,4)
            T_sMape.append(sMape_quaterly_general)
            T_Mase.append(Mase_quaterly_general) 
    # Yearly Daten
            sMape_yearly_general,Mase_yearly_general =main_predict_lstm(Dataset_yearly,6,1)
            T_sMape.append(sMape_yearly_general)
            T_Mase.append(Mase_yearly_general)
  
            p = [m[i],sMape_hourly_general,sMape_daily_general,sMape_weekly_general,
                   sMape_monthly_general,sMape_quaterly_general,sMape_yearly_general,np.mean(T_sMape),
                   Mase_hourly_general,Mase_daily_general,Mase_weekly_general,
                   Mase_monthly_general,Mase_quaterly_general,Mase_yearly_general,np.mean(T_Mase)
                  ]
            pprint(p)
            ds.loc[i] = p        
            ds=ds.round(4)
            ds.to_csv('out.csv', mode='a', header=False)
      
        
    ################################################################################
    
    print("done")

with tf.device('/gpu:1'):    
    global_prediction()

