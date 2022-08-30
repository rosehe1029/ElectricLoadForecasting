import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import  pandas as pd
import  os
from keras.models import Sequential, load_model
# load the dataset
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl 
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import numpy as np


data=pd.read_csv("transform_279675204.csv",sep=';')
data= data.iloc[3648:4370,-1]
data=data.fillna(method='ffill')
data.head(5)
dataframe = data
dataset = dataframe.values
train_size = 27*24  
trainlist = dataset[:train_size]#,:]
testlist = dataset[train_size:]#,:]

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])

    return numpy.array(dataX),numpy.array(dataY)

look_back = 1
trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))
print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
#
# create and fit the LSTM network
model = Sequential()
model.add(GRU(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
import time
T1=time.time()
history = model.fit(trainX, trainY, batch_size=64, epochs=600, validation_data=(testX,testY))
print('train用时',time.time()-T1)
def print_history(history):
    plt.plot(history.history['loss'],color='Orange',label="train loss")
    #print("history.history['loss']",len(history.history['loss']))
    plt.plot(history.history['val_loss'],color='b',label="validation loss")
    plt.title('GRU_train_validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    #plt.savefig(func_path+'/picture/trainvalloss.png')
print_history(history)
model.save(os.path.join("DATA","TJ15New" + ".h5"))
# make predictions
model = load_model(os.path.join("DATA","TJ15New" + ".h5"))
import numpy as np
trainPredict = model.predict(trainX)
T2=time.time()
testPredict = model.predict(testX)
np.savetxt('result_gru',testPredict)
print('test用时',time.time()-T2)
plt.plot(trainY)
plt.plot(trainPredict[1:])
plt.show()
plt.plot(testY)
plt.plot(testPredict[1:])
plt.show()

# 计算RMSE
rmse = np.sqrt(mean_squared_error(testY,testPredict))
r2score=r2_score(testY,testPredict)
mae=mean_absolute_error(testY,testPredict)
#mape=mean_absolute_percentage_error(test,predictions_)
print('mse',mean_squared_error(testY,testPredict))
print('rmse',rmse)
print('r2',r2score)
#print('mape',mape)
print('mae',mae)











