# -*- coding:utf-8 _*-
'''
Author: philosophylato
Date: 2022-08-30 15:04:30
LastEditors: philosophylato
LastEditTime: 2022-08-30 15:14:41
Description: your project
version: 1.0
'''
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl 
import warnings
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from math import sqrt
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# 一次性预测多个值
'''
# 文档部分参考:http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html?highlight=auto_arima
ARIMA是一个非常强大的时间序列预测模型，但是数据准备与参数调整过程非常耗时
Auto ARIMA让整个任务变得非常简单，舍去了序列平稳化，确定d值，创建ACF值和PACF图，确定p值和q值的过程

Auto ARIMA的步骤:
    1.加载数据并进行数据处理，修改成时间索引
    2.预处理数据:输入的应该是单变量，因此需要删除其他列
    3.拟合Auto ARIMA，在单变量序列上拟合模型
    4.在验证集上进行预测
    5.计算RMSE:用验证集上的预测值和实际值检查RMSE值
    
Auto-ARIMA通过进行差分测试,来确定差分d的顺序，然后在定义的start_p、max_p、start_q、max_q范围内拟合模型。
如果季节可选选项被启用，auto-ARIMA还会在进行Canova-Hansen测试以确定季节差分的最优顺序D后，寻找最优的P和Q超参数。
为了找到最好的模型,给定information_criterion auto-ARIMA优化,(‘aic’,‘aicc’,‘bic’,‘hqic’,‘oob’)
并返回ARIMA的最小值。
注意，由于平稳性问题，auto-ARIMA可能无法找到合适的收敛模型。如果是这种情况，将抛出一个ValueError，
建议在重新拟合之前使数据变稳定，或者选择一个新的顺序值范围。
非逐步的(也就是网格搜索)选择可能会很慢，特别是对于季节性数据。Hyndman和Khandakar(2008)中概述了逐步算法。
'''

#第一步：加载数据集
data=pd.read_csv("transform_279675204.csv",sep=';')
data.index=pd.to_datetime(data.datetime)
select_cols=['P']
data = data[select_cols]  
data=data.fillna(method='ffill')
data.head(5)

def arima_():
    warnings.filterwarnings(action='ignore')

    # 第二步：预处理数据-由于所给数据本身就是单变量序列，并且没有空值，因此，可以不进行这一步处理
    # 将数据分成训练集与验证集
    #val_size = 100
    #train_size = 10 * val_size
    #train, val = data[-(train_size + val_size):-val_size + 1]['data'], data[-val_size:]['data']
    # plot the data
    train=data.loc['2020-06-01 00:00:00':'2020-06-27 23:00:00',:]
    val=data.loc['2020-06-28 00:00:00':'2020-06-30 23:00:00',:]
    fig = plt.figure()
    fig.add_subplot()
    plt.figure(dpi=300,figsize=(24,8))
    plt.plot(train, 'r-', label='train_data')
    plt.plot(val, 'y-', label='val_data')
    plt.legend(loc='best')
    plt.show(block=False)

    # 第三步:buliding the model
    # 仅需要fit命令来拟合模型,而不必要选择p、d、q的组合，模型会生成AIC值和BIC值，以确定参数的最佳组合
    # AIC和 BIC是用于比较模型的评估器，这些值越低，模型就越好
    '''
    网址:http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html?highlight=auto_arima
    auto_arima部分参数解析(季节性参数未写):
        1.start_p:p的起始值，自回归(“AR”)模型的阶数(或滞后时间的数量),必须是正整数
        2.start_q:q的初始值，移动平均(MA)模型的阶数。必须是正整数。
        3.max_p:p的最大值，必须是大于或等于start_p的正整数。
        4.max_q:q的最大值，必须是一个大于start_q的正整数
        5.seasonal:是否适合季节性ARIMA。默认是正确的。注意，如果season为真，而m == 1，则season将设置为False。
        6.stationary :时间序列是否平稳，d是否为零。
        6.information_criterion：信息准则用于选择最佳的ARIMA模型。(‘aic’，‘bic’，‘hqic’，‘oob’)之一
        7.alpha：检验水平的检验显著性，默认0.05
        8.test:如果stationary为假且d为None，用来检测平稳性的单位根检验的类型。默认为‘kpss’;可设置为adf
        9.n_jobs ：网格搜索中并行拟合的模型数(逐步=False)。默认值是1，但是-1可以用来表示“尽可能多”。
        10.suppress_warnings：statsmodel中可能会抛出许多警告。如果suppress_warnings为真，那么来自ARIMA的所有警告都将被压制
        11.error_action:如果由于某种原因无法匹配ARIMA，则可以控制错误处理行为。(warn,raise,ignore,trace)
        12.max_d:d的最大值，即非季节差异的最大数量。必须是大于或等于d的正整数。
        13.trace:是否打印适合的状态。如果值为False，则不会打印任何调试信息。值为真会打印一些
    '''
    model = auto_arima(train, start_p=0, start_q=0, max_p=20, max_q=20, max_d=3,
                       seasonal=True, test='adf',m=24,
                       error_action='ignore',
                       information_criterion='bic',
                       njob=-1, trace=True, suppress_warnings=True)
    import time
    T1 = time.time()
    model.fit(train)
    print('train用时',time.time()-T1)
    #第四步:在验证集上进行预测
    T2=time.time()
    forecast1 = model.predict(n_periods=len(val))
    print('test用时',time.time()-T2)
    print(forecast1)
    forecast = pd.DataFrame(forecast1, index=val.index, columns=['prediction'])
    # calculate rmse
    rmse = np.sqrt(mean_squared_error(val, forecast))
    print('RMSE : %.4f' % rmse)
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(val, forecast))
    r2score=r2_score(val, forecast)
    mae=mean_absolute_error(val, forecast)
    #mape=mean_absolute_percentage_error(test,predictions_)
    print('mse',mean_squared_error(val, forecast))
    print('rmse',rmse)
    print('r2',r2score)
    #print('mape',mape)
    print('mae',mae)
    #plot predictions
    #fig = plt.figure()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.figure(figsize=(12,5))
    #fig.add_subplot()
    #plt.figure(dpi=300,figsize=(24,8))
    #plt.plot(train, 'r-', label='train')
    x = np.arange(72)
    plt.plot(x,val,'bo-',label='true value')
    #plt.plot(forecast,'ro-',label='predict value')
    #plt.xticks(pd.date_range('2020-6-28','2020-7-1',freq='1d'),rotation=0)
    y_offset=np.ones(72)*mae
    print(x)
    plt.errorbar(x,y=forecast1,fmt='ro-',yerr=y_offset,label="predict value")
    #plt.xticks(pd.date_range('2020-6-28','2020-7-1',freq='1d'),rotation=0)
    plt.legend(loc='best')
    plt.title('X-13-ARIMA-SEAT模型-单模型连续预测')
    plt.show()#block=False)



arima_()

    




