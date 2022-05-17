#print("hello world")
#import keyword
#print(keyword.kwlist)   
# name = "hahaah"
# print(name)
# name = 'a'
# print(name == "a")
# name = 13
# print(name == 13)
# love = 12
# print(name,love,sep="**")
# 列表 list
# list = ["a",12,True,"abc",'a']
# print(list)
# 元組 tuple
# name = ("a",14,False)
# print(name)
# 字典 
# user = {"name":"張三","age":13,"gender":"女"}
# print(user)
# 集合 set
# hobby = {"pingpang","basketball","football"} 
# print(hobby)
# password = input("請輸入密碼：")
# print(type(password))

# 百度首頁
# from urllib.request import urlopen
# url="http://www.baidu.com"
# resp = urlopen(url)
# with open("baidu.html",mode="w") as f:
#     f.write(resp.read().decode("utf-8"))
# print("over")

# age = int(input("請輸入您的年齡："))
# if age > 18:
#     print("已成年")
# else:
#     print("未成年")

# num = 0
# while num < 10:
#     print("hello world")
#     num += 1 

# name = "==fuck up=="
# print(name[0:2:1] + "--fu")
# print(name[:3] + "--fuc")
# print(name[0:] + "--fuck")
# print(name[::-1])
# print(name[-3:-1])
# print(len(name))
# print(name.count("u"))
# 大小寫轉換
# print(name.upper())
# print(name.lower())
# 大寫變小寫，小寫變大寫
# print(name.swapcase())
# 首字母大寫
# print(name.title())
# 查找字符串出現的位置
# 第一次出現的位置，找到返回下標，未找到返回-1
# print(name.find("c"))
# print(name.find("u"))
# 最後一次出現的位置
# print(name.rfind("u"))
# 與find類似，如果未找到直接報錯
# print(name.index("u"))
# 去除字符串兩端字符
# print(name.strip("="))
# 去除左邊
# print(name.lstrip("="))
# 去除右邊
# print(name.rstrip("="))

# string = "hello every one people name hi"
# 字符串切分
# print(string.split())
# 字符串合併
# text = ["hello","every","one","people","name","hi"]
# print(text)
# s = " "
# print(s.join(text))
# 檢測字符串是否全都是數字
# num = "123"
# print(num.isdigit())
# 將ascii碼轉爲對應字符
# print(chr(97))
# 將字符轉爲ascii碼值
# print(ord("a"))

# name = "張三"
# age = 13
# gender = "男"
# salary = 1234.56
# print("我是%s,今年%d歲,性別%s,工資是%.3f"%(name,age,gender,salary))
# print(f"我是{name},今年{age}歲,性別{gender},工資是{salary}")

# 遍歷列表
# list = ["張三","李四","趙名"]
# for i in list:
#     print(i)
# for index,value in enumerate(list):
#     print(index,value)
# list.append("小明")
# list.append("小麗")
# list.extend(["趙造","小李","小王"])
# print(list)
# 在指定位置插入元素
# list.insert(0,"hhh")
# print(list)
# 根據下標刪除元素
# list.pop()
# print(list)
# 根據元素名刪除元素
# list.remove("張三")
# print(list)
# 清空列表
# list.clear()
# print(list)

# 列表排序
# number = [1,2,12,23,45,3,24,46,75]
# number.sort()
# print(number)
# number.sort(reverse=True)
# print(number)
# 默認升序,返回值爲列表
# print(sorted(number,reverse=True))
# 翻轉列表
# number.reverse()
# print(number)
# 獲取列表長度
# print(len(number))
# 獲取列表中的最值
# print(max(number))
# print(min(number))
# 獲取指定元素的索引
# print(number.index(1))

# 生成1-10之間所有數
# num = list(range(1,11))
# print(num)
# 生成1 4 9 16 25
# num1 = []
# for i in range(1,6):
#     num1.append(i ** 2)
# print(num1)
# num2 = [i ** 2 for i in range(1,6)]
# print(num2)

# stu = {"name":"張三","age":"19","salary":"45556.44","gender":"n"}
# 獲取字典中元素
# print(stu["age"])
# print(stu.get("age"))
# 更改字典中元素
# stu["salary"] = 44444
# print(stu.get("salary"))
# 刪除字典中指定元素
#stu.pop("gender")
#print(stu)
# 刪除字典中最後一個元素
# stu.popitem()
# print(stu)
# 清空字典
# stu.clear()
# print(stu)
# 獲取字典長度
# print(len(stu))
# 獲取字典中所有key
# print(stu.keys())
# 獲取字典中所有value
# print(stu.values())
# 獲取所有key，value
# print(stu.items())
# 遍歷字典中所有key
# for i in stu:
#     print(i)
# 遍歷字典中所有key，value
# for k,v in stu.items():
#     print(k,v)
# 遍歷字典中所有的value
# for v in stu.values():
#     print(v)
# 一維深拷貝(對象的引用)
# num1 = [1,2,3,4,5,6,7]
# num2 = num1
# print(num1,num2)
# num2[0] = 100
# print(num1,num2)
# 一維淺拷貝
# num1 = [1,2,3,4,5,6,7]
# num2 = num1.copy()
# print(num1,num2)
# num2[0] = 100
# print(num1,num2)
# 二維淺拷貝
# num1 = [1,2,3,[3.2,3.4,3.6,3.8],5,6,7]
# num2 = num1.copy()
# print(num1,num2)
# num2[3][0] = 100
# print(num1,num2)
# 二維淺拷貝
# import copy
# num1 = [1,2,3,[3.2,3.4,3.6,3.8],5,6,7]
# num2 = copy.deepcopy(num1)
# print(num1,num2)
# num2[3][0] = 100
# print(num1,num2)

# 函數的聲明及調用
# def test():
#     for i in range(1,11):
#         print(i)
        
# a = test
# a()
# 匿名函數
# num = lambda a : a ** 2
# print(num(5))
# 回調函數
# def add(x,y):
#     print(x + y)
# def split(x,y):
#     print(x - y)
# def i(x,y,name):
#     name(x,y)
# i(3,4,add)
# i(6,4,split)    
# 閉包函數
# def demo(x):
#     y = 13
#     def ini():
#         print(x+y)
#     return ini
# fun = demo(1)
# fun()
# num = 11
# def add():
#     global num
#     num = 13
#     print(num)
# add()
# print(num)
# 篩選數據
# ages = [11,22,3,33,34,55,66,77,78]
# a = filter(lambda i : i > 30, ages)
# print(list(a))
# 處理數據
# a = map(lambda i : i + 100, ages)
# print(list(a))
# 裝飾器函數
# def outer(h):
#     def inner():
#         print("hello")
#         h()
#     return inner  
# @outer     
# def ha():
#     print("world")
# ha()
# 模塊
import random
import time
import math
import os
import datetime

# print(time.time())
# 停止五秒在執行
# time.sleep(5)
# print(random.randint(1,11))
# print(math.fabs(-111))
# print(math.sqrt(4))
# 向上取整
# print(math.ceil(3.44))
# 向下取整
# print(math.floor(3.44))
# print(math.pi)
# print(math.pow(2,3))
# print(math.factorial(5))
# 文件路徑
# print(os.getcwd())
# 當前目錄  
# print(os.curdir)
# 創建文件夾
# os.mkdir("文件名")
# 刪除文件夾,只能刪除空文件夾
# os.rmdir("文件夾路徑")
# 刪除文件
# os.remove("文件名")
# 拼接路徑
# os.path.join()
# 獲取文件大小
# os.path.getsize("文件名")
# 判斷是否是文集啊
# os.path.isfile("文件名")
# 判斷文件是否存在
# os.path.exists("文件名")
# 創建文件
# file = open("name.txt","w")
# print(random.random())
# 隨機整數
# print(random.randint(10,30))
# 隨機浮點數
# print(random.uniform(10,30))
# 從指定內容中隨機取出一個
# print(random.choice("sjd"))
# 從指定內容中隨機取出幾個
# print(random.sample("sdg",2))
# 創建日期，時間
# print(datetime.datetime(2022,12,1,12,12,12))
# 獲取當前日期時間，五天以後時間
# print(datetime.datetime.now() + datetime.timedelta(5))