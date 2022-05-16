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

# name = "==fuuck=="
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