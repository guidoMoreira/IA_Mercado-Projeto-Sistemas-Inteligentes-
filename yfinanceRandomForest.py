import yfinance as yf
import pandas as pd
import os

#pip3 install yfinance
#pip3 install plotly
#pip3 install -U scikit-learn


ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2023-05-12"

data = yf.download(ticker,start= start_date,end=end_date)

#print(data)

df = pd.DataFrame(data)

#print(df.head())

#df.info()

df['date']= pd.to_datetime(df.index)
#print(df.head())
print(df.tail())


#Parte 2 Visualziar
'''
import plotly.graph_objects as go

fig = go.Figure(data=[go.Candlestick(x=df['date'],
		open=df['Open'],
		high=df['High'],
		low=df['Low'],
		close=df['Close'])])

fig.update_layout(
	title='Stock price char AAPL',
	yaxis_title='Price ($)',
	xaxis_rangeslider_visible=False)

fig.show()
'''

#Parte 3 Criar modelo com random forest

auxdate = df['date']

df.drop(['date','Volume'], axis=1,inplace=True)
df.reset_index(drop=True,inplace=True)
#df.plot.line(y="Close",use_index=True)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = df[['Open','Close','High','Low','Adj Close']]#inputs
y = df['Close']#Alvo


X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=242,shuffle=False)
#random forest
rf = RandomForestRegressor(n_estimators=60,random_state=3)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

import matplotlib.pyplot as plt

print(len(list(df['Close'])))

print(y_pred)
ls = list(y_test)
print(ls)

plt.plot(list(y_train)+ls)#df['Close'])
plt.plot(list(y_train)+list(y_pred))
plt.show()


#Avaliar modelo com erro de média quadrada
mse = mean_squared_error(y_test,y_pred)


print("Erro: ",mse,"%")


LancesCerto = 0
LancePerdidos = 0
listv = df['Close'].values.tolist()

listv = listv[len(y_pred):]

for i in range(1,len(y_pred)):
  closeat = listv[i] # atual
  closeant = listv[i-1]# anterior
  prediat = y_pred[i] #- rmse# previsão pro atual
  prediant = y_pred[i-1]# previsão anterior
  if round(closeat, 5)  >= round(closeant, 5):
    if round(prediat, 5) >= round(prediant,5):#Previsão que subiria CERTA
      LancesCerto+=1
    else:#Previsão que deceria ou manteria Errada
      LancePerdidos+=1

  #desceu ou não mudou
  else:     # elif closeat < closeant
    if round(prediat, 5) > round(prediant,5):#Previsão que subiria ERRADA
      LancePerdidos += 1
    else: #Previsão que desceria ou não mudaria CERTA
      LancesCerto +=1

print("Decisões Acertadas: ", LancesCerto)
print("Decisões Erradas: ", LancePerdidos)
print("Porcentagem Acerto {:.2f} %".format(100*LancesCerto/(LancePerdidos+LancesCerto)))




'''
fig.update_layout(
	title='Stock price char AAPL',
	yaxis_title='Price ($)',
	xaxis_rangeslider_visible=False)

fig.show()
'''

#Parte 4 usar modelo em outros dados

import numpy as np

new_data = np.array([[173.850006,174.589996,172.169998,173.75000,173.750000]])

predict_price = rf.predict(new_data)

print("Preço previsto: ",predict_price[0])

