key="19b35158f6d02dd1d71eaf411be748c58133362c"
import os
config={}
import tensorflow as tf


config['api_key']="19b35158f6d02dd1d71eaf411be748c58133362c"
from tiingo import TiingoClient
clinet= TiingoClient(config)

ticker=clinet.get_dataframe('AAPL',startDate='2002-05-26')


#import pandas_datareader.data as web
#df=web.get_data_tiingo('AAPL',api_key=key)

df=ticker.reset_index()['close']

import matplotlib.pyplot as plt
plt.plot(df)

import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df=scaler.fit_transform(np.array(df).reshape(-1,1))

training_size=int(len(df)*0.65)
test_size=len(df)-training_size
train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]

import numpy
def create_databse(dataset,time_step=1):
    datax,datay=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        datax.append(a)
        datay.append(dataset[i+time_step,0])
    return numpy.array(datax),numpy.array(datay)

time_step=100
x_train,y_train=create_databse(train_data,time_step)
x_test,y_test=create_databse(test_data,time_step)


print(x_train)

x_test.shape
y_test.shape

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#creating stacked lstm

from keras.models import Sequential
from keras.layers import Dense,LSTM

model=Sequential()
model.add(LSTM(50, return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error,accuracy_score

math.sqrt(mean_squared_error(y_train, train_predict))

math.sqrt(mean_squared_error(y_test, test_predict))



##plot
#shift train predictions for plotting
look_back=100
trainpredictplot=numpy.empty_like(df)
trainpredictplot[:,:]=np.nan
trainpredictplot[look_back:len(train_predict)+look_back,:]=train_predict
#shift test prediction for plotting
testpredictplot=numpy.empty_like(df)
testpredictplot[:,:]=numpy.nan
testpredictplot[len(train_predict)+(look_back*2)+1:len(df)-1,:]=test_predict
# plot baseline and predictions

plt.plot(scaler.inverse_transform(df))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.show
## lets predict next 30 days 
len(test_data)
1928-100

x_input=test_data[1828:].reshape(1,-1)
x_input.shape

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

temp_input

lst_output=[]
n_steps=100
i=0 
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{}day input{}".format(i, x_input))
        x_input=x_input.reshape(1, -1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{}day input{}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input =x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)        

day_new=np.arange(1,101 )
day_pred=np.arange(101,131 )
len(df)


df3=df.tolist()
df3.extend(lst_output)
plt.plot(day_new,scaler.inverse_transform(test_data[1828:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

next30days=scaler.inverse_transform(lst_output)
