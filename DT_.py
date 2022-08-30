import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###########1.数据生成部分##########
import numpy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pylab as pl 

from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
data=pd.read_csv("transform_279675204.csv",sep=';')
data= data.iloc[3648:4370,-1]
 
data=data.fillna(method='ffill')
data.head(5)
dataset = data.values
train_size = 720-72     #int(len(dataset) * 0.65)
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
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
#trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))
print(trainX.shape,trainY.shape)
print(testX.shape,testY.shape)
#
def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / 120
    else:
        return None

n_train = trainX.shape[0]  # 样本个数
x_train=trainX
x_test = testX
y_train =trainY
y_test=testY

###########2.回归部分##########
def try_different_method(model):
    model.fit(x_train,y_train)
    
    score = model.score(x_test, y_test)
   
    result = model.predict(x_test)
   
    print('mse', get_mse(result, y_test) )
    print(score)
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test,result))
    r2score=r2_score(y_test,result)
    mae=mean_absolute_error(y_test,result)
    #mape=mean_absolute_percentage_error(test,predictions_)
    print('mse',mean_squared_error(y_test,result))
    print('rmse',rmse)
    print('r2',r2score)
    #print('mape',mape)
    print('mae',mae)
    np.savetxt('result_DecisionTree',result)
    #np.savetxt('result/result_LinearRegression', result)
    #np.savetxt('result0/result_SVM', result)
    #np.savetxt('result_KNeighborsRegressor', result)
    #np.savetxt('result_RandomForestRegressor', result)
    #np.savetxt('result0/result_GradientBoostingRegressor', result)
    #np.savetxt('result0/result_AdaBoostRegressor', result)
    #np.savetxt('result0/result_GradientBoostingRegressor', result)
    #np.savetxt('result0/result_BaggingRegressor', result)
    #np.savetxt('result0/result_ExtraTreeRegressor', result)
    #np.savetxt('result0/result_ARDRegressor', result)
    #np.savetxt('result0/result_BayesianRidge', result)
    #np.savetxt('result0/result_TheilSenRegressor', result0)
    #np.savetxt('result0/result_RANSACRegressor', result)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.figure(figsize=(12,5))
    plt.plot(y_test,'bo-',label='true value')
    plt.plot(result,'ro-',label='predict value')
    #plt.xticks(pd.date_range('2020-6-28','2020-7-1',freq='1d'))
    #plt.title('决策树模型')
    plt.legend(loc='best')
    plt.show()

###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm
model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors=20)
####3.5随机森林回归####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10ARD贝叶斯ARD回归
model_ARDRegression = linear_model.ARDRegression()
####3.11BayesianRidge贝叶斯岭回归
model_BayesianRidge = linear_model.BayesianRidge()
####3.12TheilSen泰尔森估算
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13RANSAC随机抽样一致性算法
model_RANSACRegressor = linear_model.RANSACRegressor()

###########4.具体方法调用部分##########
#决策树回归结果
try_different_method(model_DecisionTreeRegressor)
#线性回归结果
#try_different_method(model_LinearRegression)
#SVM回归结果
#try_different_method(model_SVR)
#KNN回归结果
#try_different_method(model_KNeighborsRegressor)
#随机森林回归结果
#try_different_method(model_RandomForestRegressor)
#Adaboost回归结果
#try_different_method(model_AdaBoostRegressor)
#GBRT回归结果
#try_different_method(model_GradientBoostingRegressor)
#Bagging回归结果
#try_different_method(model_BaggingRegressor)
#极端随机树回归结果
#try_different_method(model_ExtraTreeRegressor)
#贝叶斯ARD回归结果
#try_different_method(model_ARDRegression)
#贝叶斯岭回归结果
#try_different_method(model_BayesianRidge)
#泰尔森估算回归结果
#try_different_method(model_TheilSenRegressor)
#随机抽样一致性算法
#try_different_method(model_RANSACRegressor)


