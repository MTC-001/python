import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras import models
import keras.backend as K
from sklearn.metrics import r2_score

(x_train,y_train),(x_test,y_test) = boston_housing.load_data()
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_train[0])
print(x_test[0])
n_hidden_1 = 64
n_hidden_2 = 64
n_input = 13
n_classes = 1
training_epochs = 200
batch_size = 10
model = models.Sequential()
model.add(Dense(n_hidden_1,activation='relu',input_dim=n_input))
model.add(Dense(n_hidden_2,activation='relu'))
model.add(Dense(n_classes))
def r2(y_true,y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
model.compile(loss='mse',optimizer='rmsprop',metrics=['mae',r2])
history = model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs)
pred_y_test = model.predict(x_test)
print(pred_y_test)
pred_acc = r2_score(y_test,pred_y_test)
print('pred_acc',pred_acc)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16,8), dpi=160)
plt.plot(range(len(y_test)), y_test, ls='-.',lw=2,c='r',label='真实值')
plt.plot(range(len(pred_y_test)), pred_y_test, ls='-',lw=2,c='b',label='预测值')
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.xlabel('number')
plt.ylabel('房价')
plt.show()