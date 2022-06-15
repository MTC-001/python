import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pandas import read_csv
 
 
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT ', 'MEDV']
data = pd.read_csv(filename,names= names, delim_whitespace=True)
dataset = data.values
   
data.isnull().any().sum()
pd.plotting.scatter_matrix(data, alpha=0.7, figsize=(10,10), diagonal='kde')
plt.show()
 
corr = data.corr()
plt.matshow(data.corr())
plt.show()
 
x = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT ']]
y = data[['MEDV']]
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
SelectKBest = SelectKBest(f_regression, k=3)
bestFeature = SelectKBest.fit_transform(x,y)
SelectKBest.get_support()
x.columns[SelectKBest.get_support()]
features = data[['RM', 'PTRATIO', 'LSTAT ']]
pd.plotting.scatter_matrix(features, alpha=0.7, figsize=(6,6), diagonal='hist')
plt.show()
 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for feature in features.columns:
    features['标准化'+feature] = scaler.fit_transform(features[[feature]])
 
#散点可视化，查看特征归一化后的数据
font={
      'family':'SimHei'
      }
matplotlib.rc('font', **font)
pd.plotting.scatter_matrix(features[['标准化RM', '标准化PTRATIO', '标准化LSTAT ']], alpha=0.7, figsize=(6,6), diagonal='hist')
plt.show()
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features[['标准化RM', '标准化PTRATIO', '标准化LSTAT ']], y, test_size=0.3,random_state=33)
 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
 
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr_predict = cross_val_predict(lr,x_train, y_train, cv=5)
lr_score = cross_val_score(lr, x_train, y_train, cv=5)
lr_meanscore = lr_score.mean()
 
#SVR
from sklearn.svm import SVR
linear_svr = SVR(kernel = 'linear')
linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
linear_svr_meanscore = linear_svr_score.mean()
 
poly_svr = SVR(kernel = 'poly')
poly_svr_predict = cross_val_predict(poly_svr, x_train, y_train, cv=5)
poly_svr_score = cross_val_score(poly_svr, x_train, y_train, cv=5)
poly_svr_meanscore = poly_svr_score.mean()
 
rbf_svr = SVR(kernel = 'rbf')
rbf_svr_predict = cross_val_predict(rbf_svr, x_train, y_train, cv=5)
rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
rbf_svr_meanscore = rbf_svr_score.mean()
 
from sklearn.neighbors import KNeighborsRegressor
score=[]
for n_neighbors in range(1,21):
    knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
    knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
    knn_score = cross_val_score(knn, x_train, y_train, cv=5)
    knn_meanscore = knn_score.mean()
    score.append(knn_meanscore)
plt.plot(score)
plt.xlabel('n-neighbors')
plt.ylabel('mean-score')
plt.show()
 
n_neighbors=2
knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
knn_predict = cross_val_predict(knn, x_train, y_train, cv=5)
knn_score = cross_val_score(knn, x_train, y_train, cv=5)
knn_meanscore = knn_score.mean()
 
#Decision Tree
from sklearn.tree import DecisionTreeRegressor
score=[]
for n in range(1,11):
    dtr = DecisionTreeRegressor(max_depth = n)
    dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
    dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
    dtr_meanscore = dtr_score.mean()
    score.append(dtr_meanscore)
plt.plot(np.linspace(1,10,10), score)
plt.xlabel('max_depth')
plt.ylabel('mean-score')
plt.show()
 
n=4
dtr = DecisionTreeRegressor(max_depth = n)
dtr_predict = cross_val_predict(dtr, x_train, y_train, cv=5)
dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
dtr_meanscore = dtr_score.mean()
 
 
evaluating = {
        'lr':lr_score,
        'linear_svr':linear_svr_score,
        'poly_svr':poly_svr_score,
        'rbf_svr':rbf_svr_score,
        'knn':knn_score,
        'dtr':dtr_score
        }
evaluating = pd.DataFrame(evaluating)
 
evaluating.plot.kde(alpha=0.6,figsize=(8,7))
plt.show()
 
evaluating.hist(color='k',alpha=0.6,figsize=(8,7))
plt.show()
 
 
lSVR_score=[]
for i in [1,10,1e2,1e3,1e4]:
    linear_svr = SVR(kernel = 'linear', C=i)
    linear_svr_predict = cross_val_predict(linear_svr, x_train, y_train, cv=5)
    linear_svr_score = cross_val_score(linear_svr, x_train, y_train, cv=5)
    linear_svr_meanscore = linear_svr_score.mean()
    lSVR_score.append(linear_svr_meanscore)
plt.plot(lSVR_score)
plt.show()
 
 
 
 
 
 
 
#模型预测
#rbf
rbf_svr.fit(x_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)
rbf_svr_y_predict_score=rbf_svr.score(x_test, y_test)
#KNN
knn.fit(x_train,y_train)
knn_y_predict = knn.predict(x_test)
knn_y_predict_score = knn.score(x_test, y_test)
#poly_svr
poly_svr.fit(x_train,y_train)
poly_svr_y_predict = poly_svr.predict(x_test)
poly_svr_y_predict_score = poly_svr.score(x_test, y_test)
#dtr
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)
dtr_y_predict_score = dtr.score(x_test, y_test)
#lr
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
lr_y_predict_score = lr.score(x_test, y_test)
#linear_svr
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)
linear_svr_y_predict_score = linear_svr.score(x_test, y_test)
predict_score = {
        'lr':lr_y_predict_score,
        'linear_svr':linear_svr_y_predict_score,
        'poly_svr':poly_svr_y_predict_score,
        'rbf_svr':rbf_svr_y_predict_score,
        'knn':knn_y_predict_score,
        'dtr':dtr_y_predict_score
        }
predict_score = pd.DataFrame(predict_score, index=['score']).transpose()
print(predict_score.sort_values(by='score',ascending = False))
print(x.columns[SelectKBest.get_support()])
 
#对各个模型的预测值整理
plt.scatter(np.linspace(0,151,152),y_test, label='predict data')
labelname=[
        'rbf_svr_y_predict',
        'knn_y_predict',
        'poly_svr_y_predict',
        'dtr_y_predict',
        'lr_y_predict',
        'linear_svr_y_predict']
y_predict={
        'rbf_svr_y_predict':rbf_svr_y_predict,
        'knn_y_predict':knn_y_predict[:,0],
        'poly_svr_y_predict':poly_svr_y_predict,
        'dtr_y_predict':dtr_y_predict,
        'lr_y_predict':lr_y_predict[:,0],
        'linear_svr_y_predict':linear_svr_y_predict
        }
y_predict=pd.DataFrame(y_predict)
for name in labelname:
    plt.plot(y_predict[name],label=name)
plt.xlabel('predict data index')
plt.ylabel('target')
plt.legend()
plt.show()
 
 
for column in corr.columns:
    inxs = (abs(corr[column])<1) & (abs(corr[column])>0.6)
    for inx in corr.index[inxs]:
        print(column+'<-->'+inx)
        
corr['MEDV'].abs().sort_values(ascending=False)
 
 
#线性相关可视化
import matplotlib.pyplot as plt
import numpy as np
 
def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = i
        aa[14-i,i] = -i
    return aa
 
# Display matrix
plt.matshow(samplemat((15, 15)))
plt.show()
 
knn_meanscore = []
dtr_meanscore = []
rbf_svr_meanscore = []
features=data[['RM','PTRATIO','LSTAT ','TAX','ZN','B','CHAS','INDUS','NOX','RAD','AGE','CRIM','DIS']]
 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
for feature in features.columns:
    features['标准化'+feature] = scaler.fit_transform(features[[feature]])
 
 
new = features[['标准化LSTAT ','标准化PTRATIO','标准化TAX','标准化ZN','标准化B','标准化CHAS','标准化RM','标准化INDUS','标准化NOX','标准化RAD','标准化AGE','标准化CRIM','标准化DIS']]
 
for m in range(1,14):
	print('feature num: %d\r\n' %m)
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import f_regression
	selectKBest = SelectKBest(f_regression, k=int(m))
	bestFeature = selectKBest.fit_transform(new,y)   
	xx = new[new.columns[selectKBest.get_support()]]
	print("features' name: ", xx.columns.tolist()) 
	x_train, x_test, y_train, y_test = train_test_split(xx,y,test_size=0.3,random_state=33)
    #knn
	n_neighbors=2
	knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
	knn_score = cross_val_score(knn, x_train, y_train, cv=5)
	knn_meanscore.append(knn_score.mean())
    #Decision Tree
	dtr = DecisionTreeRegressor(max_depth = 4)
	dtr_score = cross_val_score(dtr, x_train, y_train, cv=5)
	dtr_meanscore.append(dtr_score.mean())
    #rbf
	rbf_svr = SVR(kernel = 'rbf',C=100,gamma=0.5)
	rbf_svr_score = cross_val_score(rbf_svr, x_train, y_train, cv=5)
	rbf_svr_meanscore.append(rbf_svr_score.mean())
 
evaluating = {
            'rbf_svr':rbf_svr_meanscore,
            'knn':knn_meanscore,
            'dtr':dtr_meanscore
            }
evaluating = pd.DataFrame(evaluating)    
 
 
evaluating.plot()
plt.xticks(evaluating.index,range(1, 13))
plt.xlabel("features' number")
plt.ylabel('Model score')
 
plt.show()
 
 
selectKBest = SelectKBest(f_regression, k=6)
bestFeature = selectKBest.fit_transform(new,y)   
xx = new[new.columns[selectKBest.get_support()]]
print("features' name: ", xx.columns.tolist()) 
#knn
n_neighbors=2
knn = KNeighborsRegressor(n_neighbors, weights = 'uniform' )
knn.fit(x_train, y_train)
knn_y_predict = knn.predict(x_test)
knn_y_predict_score = knn.score(x_test, y_test)
#Decision Tree
dtr = DecisionTreeRegressor(max_depth = 4)
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)
dtr_y_predict_score = dtr.score(x_test, y_test)
#rbf
rbf_svr = SVR(kernel = 'rbf',C=100,gamma=0.5)
rbf_svr.fit(x_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)
rbf_svr_y_predict_score=rbf_svr.score(x_test, y_test)
 
 
predict_score = {
        'rbf_svr':rbf_svr_y_predict_score,
        'knn':knn_y_predict_score,
        'dtr':dtr_y_predict_score
        }
predict_score = pd.DataFrame(predict_score, index=['score']).transpose()
print(predict_score.sort_values(by='score',ascending = False))