import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv",header=1,names=['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '品种'])
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("200511616-佟蕾")
parallel_coordinates(iris,"品种")
plt.show()
