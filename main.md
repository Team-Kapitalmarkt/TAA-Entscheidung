import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

Zeitintervall = 30
ts = ["TAA", "DAX", "Stoxx600"]
a = []
df = pd.read_excel("TAA_Entscheidungen.xlsx",sheet_name=[ts[0], ts[1], ts[2]])

TAA_Date = pd.DataFrame(index=df[ts[0]].iloc[:,0])
Anzahl_TAA = len(TAA_Date)

for i in range(0,len(ts)):
    a.append(pd.DataFrame(df[ts[i]].values[:,1],index=df[ts[i]].iloc[:,0]))
    
a[0][a[0]=='einfach untergewichten'] = -1
a[0][a[0] == 'einfach übergewichten'] = 1
a[0][a[0] == 'neutral'] = 0

for i in range(0,len(ts)):
    a[i].columns = [ts[i]]

y_1 = pd.concat([a[2]], axis=1)

TAA_Date1 = TAA_Date.index  + datetime.timedelta(days=0)
TAA_Date2 = TAA_Date.index  + datetime.timedelta(days=Zeitintervall)

y_1.index = pd.to_datetime(y_1.index)    

for i in range(0, len(TAA_Date1)-1):
    y_3 = y_1[(y_1.index > TAA_Date1[i]) & (y_1.index < TAA_Date2[i])]
    y_len = len(y_3)
    
    x = pd.to_numeric(y_3[ts[2]].to_numpy()).reshape(y_len,1)
    x = pd.Index.to_numpy

    X = np.linspace(0, y_len-1, num=y_len).reshape(y_len,1)
    Y = pd.to_numeric(y_3[ts[2]].to_numpy()).reshape(y_len,1)
    model = LinearRegression().fit(X, Y) 
    x_new = np.linspace(0, y_len, 100)
    y_new = model.predict(x_new[:, np.newaxis])

    plt.figure(figsize=(4, 3))
    ax = plt.axes()
    ax.plot(x_new,y_new)
    ax.plot(X, Y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('tight')
    plt.show()

    print('Score: ',model.score(X, Y))
    print('Coefficients: ', model.coef_)
    print('Mean squared error: ', mean_squared_error(X,Y))
    print('Coefficient of determination: ', r2_score(X,Y))
