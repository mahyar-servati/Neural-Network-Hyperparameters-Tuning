# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:24:28 2023

@author: Mahyar Servati
"""
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from mealpy.swarm_based import MFO, ALO
import pandas as pd
from keras import optimizers
import time
import os
from keras.layers import RNN, SimpleRNN
from keras.optimizers import Adam
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import MaxPooling1D

start_time = time.time()

Model = 'BiLSTM'
season = 'spring'


Save_dir = f"ALO_{Model}"

if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def xy_split(df, target):
    y_clmns=[target]
    x_clmns=df.columns.tolist()
    remove_clmns=[target]
    for arg in remove_clmns:
        x_clmns.remove(arg)
    X=df[x_clmns]
    y=df[y_clmns]
    return X, y    


df = pd.read_csv("../DNI.Jiangsu.csv")#, index_col='Time')
# df.index = list(map(lambda x:x.replace("T", " "),df.index))
df['Time'] = list(map(lambda x:x.replace("T", " "),df['Time']))
df.set_index(pd.to_datetime(df['Time']), inplace=True)
df.drop(['Time'], axis=1, inplace=True)

if season == 'spring':
    df2 = df[:2160]
elif season == 'summer':
    df2 = df[2160:4344]
elif season == 'fall':
    df2 = df[4344:6552]
elif season == 'winter':
    df2 = df[6552:]
data = df2.copy()

data.DNI.interpolate(inplace = True)

X, y = xy_split(data, 'DNI')

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=1)

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# scaler = MinMaxScaler()
# scaled_seq = scaler.fit_transform(np.reshape(np.array(df), (-1,1)))
# n_steps = 30
# X_train, y_train = split_sequence(scaled_seq[0:int(len(scaled_seq)*0.8)], n_steps)
# X_test, y_test = split_sequence(scaled_seq[int(len(scaled_seq)*0.8):], n_steps)

OPT_ENCODER = LabelEncoder()
OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  

WOI_ENCODER = LabelEncoder()
WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

ACT_ENCODER = LabelEncoder()
ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])
 
def decode_solution(solution):
    batch_size = 2**int(solution[0])
    epoch = 100 * int(solution[1])
    opt_integer = int(solution[2])
    opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    # opt = 'Adam'

    learning_rate = solution[2]
    network_weight_initial_integer = int(solution[4])
    network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    # network_weight_initial = 'normal'
    act_integer = int(solution[5])
    activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    # activation = 'relu'
    n_hidden_units = 2**int(solution[3])
    return [batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units]



def objective_function(solution): 
    batch_size, epoch, opt, learning_rate, network_weight_initial, Activation, n_hidden_units = decode_solution(solution)
    if Model == 'LSTM':
        model = Sequential()
        model.add(LSTM(units = n_hidden_units, activation=Activation,
                       kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2)) 
        model.add(LSTM(units = n_hidden_units)) 
        model.add(Dropout(0.2))
        model.add(Dense(units = 1)) 
    elif Model == 'BiLSTM':
        model = Sequential()
        model.add(Bidirectional(LSTM(units = n_hidden_units, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2]))))
        model.add(Dense(1))
    elif Model == 'CNN':
        model = Sequential()
        model.add(Conv1D(filters=n_hidden_units, kernel_size=2, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
    elif Model == 'RNN':
        model = Sequential()
        model.add(SimpleRNN(n_hidden_units, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = False))
        model.add(Dense(1, activation='relu'))

    optimizer = getattr(optimizers, opt)(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)
    yhat = model(X_test)
    fitness = mean_absolute_error(y_test, yhat)
    return fitness




hyper_list = ['batch_size', 'epoch', 'opt', 'learning_rate', 'network_weight_initial', 'activation', 'n_hidden_units']
# LB = [2, 1, 0.01, 2]
# UB = [4.99, 10.99, 1.0, 6.99]

LB = [1, 7, 0, 0.01, 0, 0, 2]
UB = [3.99, 20.99, 6.99, 0.5, 7.99, 7.99, 10]
      
problem = {
    'fit_func' : objective_function,
    'lb' : LB,
    'ub' : UB,
    'minmax' : 'min',
    'verbose' : True,
    }

Epoch = 100 #should be in [2, 10000]
Pop_Size = 10 #should be in [10, 10000]

optmodel = ALO.BaseALO(problem, epoch=Epoch, pop_size=Pop_Size)
optmodel.solve()

bests = optmodel.solution[0]



def pred_model(best): 
    batch_size, epoch, opt, learning_rate, network_weight_initial, Activation, n_hidden_units = decode_solution(best)
    if Model == 'LSTM':
        model = Sequential()
        model.add(LSTM(units = n_hidden_units, activation=Activation,
                       kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2)) 
        model.add(LSTM(units = n_hidden_units)) 
        model.add(Dropout(0.2))
        model.add(Dense(units = 1)) 
    elif Model == 'BiLSTM':
        model = Sequential()
        model.add(Bidirectional(LSTM(units = n_hidden_units, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2]))))
        model.add(Dense(1))
    elif Model == 'CNN':
        model = Sequential()
        model.add(Conv1D(filters=n_hidden_units, kernel_size=2, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        
    elif Model == 'RNN':
        model = Sequential()
        model.add(SimpleRNN(n_hidden_units, activation=Activation,
                                      kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = False))
        model.add(Dense(1, activation='relu'))
    optimizer = getattr(optimizers, opt)(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, np.array(y_train).reshape(-1,1), epochs=epoch, batch_size=batch_size, verbose=0)
    pred = model.predict(X_test)
    return pred

pred = pred_model(bests)

Best_Hyper =  pd.DataFrame(bests, columns=['Best Hyperparameters'], index=hyper_list)

with pd.ExcelWriter(f'{Save_dir}/Best_Hyperparameters.xlsx' ,engine='xlsxwriter') as Writer:
    Best_Hyper.to_excel(Writer, index=True, startrow=0, startcol=0)

actual_pred = y_test.copy()
actual_pred.rename(columns={'DNI':'Actual'}, inplace=True)
actual_pred['Predict'] = pred.flatten()

with pd.ExcelWriter(f'{Save_dir}/Actual_Predict.xlsx' ,engine='xlsxwriter') as Writer:
    actual_pred.to_excel(Writer, index=True, startrow=0, startcol=0)

population = pd.DataFrame(np.array([(optmodel.history.list_population[i][j][0]) for i in range(Epoch) for j in range(Pop_Size)]), columns=hyper_list)
population['MSE'] = np.array([optmodel.history.list_population[i][j][1][0] for i in range(Epoch) for j in range(Pop_Size)])
data_out = pd.DataFrame(np.array([(optmodel.history.list_global_best[i][0]) for i in range(Epoch)]), columns=hyper_list)
data_out['diversity'] =(pd.DataFrame(optmodel.history.list_diversity))
data_out['exploration'] =(pd.DataFrame(optmodel.history.list_exploration))
data_out['exploitation'] =(pd.DataFrame(optmodel.history.list_exploitation))
data_out['epoch_time'] =(pd.DataFrame(optmodel.history.list_epoch_time))
data_out['MSE'] =np.array([optmodel.history.list_global_best[i][1][0] for i in range(Epoch)])

with pd.ExcelWriter(f'{Save_dir}/Optimizing_data_out.xlsx' ,engine='xlsxwriter') as Writer:
    data_out.to_excel(Writer, index=True, startrow=0, startcol=0)
with pd.ExcelWriter(f'{Save_dir}/Optimizing_population.xlsx' ,engine='xlsxwriter') as Writer:
    population.to_excel(Writer, index=True, startrow=0, startcol=0)


end_time = time.time()
calculation_time = pd.DataFrame([round(end_time-start_time, 2)], columns=['Total_calculation_time'])
with pd.ExcelWriter(f'{Save_dir}/Total_calculation_time.xlsx' ,engine='xlsxwriter') as Writer:
    calculation_time.to_excel(Writer, index=True, startrow=0, startcol=0)
print('Optimazing and prediction done in', round(end_time-start_time, 2), "secound")





# start_time = time.time()

# model = Sequential()
# model.add(Bidirectional(LSTM(units = 32, activation='relu',
#                               kernel_initializer='normal', input_shape=(X_train.shape[1], X_train.shape[2]))))
# model.add(Dense(1))

# model = Sequential()
# model.add(SimpleRNN(32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = False))
# model.add(Dense(1, activation='relu'))

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1))


# model = Sequential()
# model.add(LSTM(units = 32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dense(1))

# optimizer = getattr(optimizers, 'Adam')(lr=0.117)
# model.compile(optimizer=optimizer, loss='mse')
# model.fit(X_train, np.array(y_train).reshape(-1,1), epochs=200, batch_size=8, verbose=1)
# pred = model.predict(X_test)

# end_time = time.time()

# print(round(end_time-start_time, 2))

 
