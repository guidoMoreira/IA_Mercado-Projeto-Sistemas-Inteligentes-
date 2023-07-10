import MetaTrader5 as mt5
import tensorflow as tf
import math
import numpy as np
from keras.optimizers import Adam
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
from _datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'

if (mt5.initialize()):  #pegar dados do Metatrader
  print("ok")
else:
  print("falha")
a = mt5.terminal_info()
ativo = "USDJPY"  #ativo do dolar e japao
mt5.symbol_select(ativo, True)
f = mt5.copy_rates_from_pos(ativo, mt5.TIMEFRAME_M5, 0, 3000)
g = pd.DataFrame(f)
#print(g)
g['time'] = pd.to_datetime(g['time'], unit='s')
g['close'].plot

date = datetime(2023, 6, 2)  # tempo para analise
flag = mt5.COPY_TICKS_ALL  #base de dados
dados = mt5.copy_ticks_from(ativo, date, 10,flag)  #analise de dados referente ao ativo no tempo determidado
df = pd.DataFrame(dados)
mt5.shutdown()  # fecha o sistema

#plt.plot(g['time'],g['close'])
#plt.show()
#print(g.columns)

#Tamanho de cada input para a rede neural
Mrange = 30
#------ Modelar os dados para treinamento
threshold = 0.6
data = g.filter(['close'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*threshold)

scaler = MinMaxScaler(feature_range=(-0.5,0.5))

scale_data = scaler.fit_transform(dataset)

#criar dado de treino
train_data = scale_data[0:training_data_len,:]
x_train = []
y_train = []

for i in range(Mrange,len(train_data)):
  x_train.append(train_data[i-Mrange:i, 0])
  y_train.append(train_data[i, 0])

  #if(i<= Mrange):
#print("->",x_train)
#print("->",y_train)
    #print(y_train)

x_train,y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],Mrange,1))

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="sgd",loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)
#print(y_train)

#testando modelo
test_data = scale_data[training_data_len - Mrange: , :]
x_test = []
y_test = dataset[training_data_len:,:]
#print(test_data)
for i in range(Mrange,len(test_data)):
  x_test.append(test_data[i-Mrange:i,0])


x_test = np.array(x_test)
#print(x_test.shape)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Gambiarra para prever o proximo valor apartir do valor previsto
prediction = []

'''for i in range(0,100):#len(x_test)):
  print("Etapa ",i)
  x_test1 = [test_data[0:Mrange,0]]
  #print(x_test1)
  if i >1:
    for h in range(0,Mrange):
      if h >= len(prediction)-1:
        break
      auxl = list(x_test1[0])
      auxl.pop(0)
      auxl.append(prediction[len(prediction) - (h + 1)])
      x_test1[0] = tf.convert_to_tensor(auxl)
  x_test1 = np.array(x_test1)
  x_test1 = np.reshape(x_test1, (x_test1.shape[0],  x_test1.shape[1], 1))
  prediction.append(model.predict(x_test1)[0][0])
  print(prediction[i])

'''
'''x_test1 = [test_data[0:60,0]]
x_test1 = np.array(x_test1)
x_test1 = np.reshape(x_test1, (x_test1.shape[0], x_test1.shape[1], 1))


prediction = model.predict(x_test1)'''

prediction = model.predict(x_test)
#print("->",prediction)
prediction = scaler.inverse_transform(prediction)
#print("2->",prediction)

rmse = np.sqrt(np.mean(prediction - y_test)**2)
print("Erro Quandratico médio {:.2f}%".format(rmse*100))

train = data[:training_data_len]
valid = data[training_data_len:]
#print(valid)
valid['Predictions'] = prediction


LancesCerto = 0
LancePerdidos = 0
listv = valid.values.tolist()

for i in range(1,len(prediction)):
  closeat = listv[i][0] # atual
  closeant = listv[i-1][0]# anterior
  prediat = listv[i][1] - rmse*(listv[i][1])# previsão pro atual consertada
  prediant = listv[i-1][1]- rmse*(listv[i][1])# previsão anterior consertada
  #Subiu
  #print("--c>",closeat,closeant)
  #print("--p>", prediat, prediant)
  # fechamento atual > Fechamento anterior = Subida
  if round(closeat,5) >= round(closeant, 5):
    if round(prediat, 5) >= round(prediant,5):#Previsão que subiria CERTA
      LancesCerto+=1
    else:#Previsão que deceria ou manteria Errada
      LancePerdidos+=1

  #desceu
  else:     # elif closeat < closeant
       #Fechamento Atual < Fechamento anterior = Descida
       #previsao atual < previsao anterior
    if round(prediat, 5) <= round(prediant,5):#Previsão que desceria ou não mudaria CERTA
      LancesCerto += 1
    else: #Previsão que subiria ERRADA
      LancePerdidos += 1

print("Decisões Acertadas: ", LancesCerto)
print("Decisões Erradas: ", LancePerdidos)
print("Porcentagem Acerto {:.2f} %".format(100*LancesCerto/(LancePerdidos+LancesCerto)))

plt.figure(figsize=(16,8))
plt.title('Modelo')
plt.xlabel('Data', fontsize=18)
plt.ylabel('Preço Fechamento', fontsize=18)

plt.plot(train['close'])
plt.plot(valid[['close','Predictions']])
#plt.plot(list(train['close']))

#plt.plot(list(train['close'])+list(prediction[0]))
#plt.plot(data[:1350])
plt.legend(['Treino','val','Previsão'], loc= 'lower right')
plt.text(0.8,0.2,"Erro Quandratico médio {:.2f}%".format(rmse*100), transform=plt.gca().transAxes)
plt.text(0.8,0.3,"Decisões Acertadas: {:} / {:}" .format(LancesCerto,(LancesCerto+LancePerdidos)), transform=plt.gca().transAxes)
plt.text(0.8,0.25,"Porcentagem Acerto {:.2f} %".format(100*LancesCerto/(LancePerdidos+LancesCerto)), transform=plt.gca().transAxes)

plt.show()

