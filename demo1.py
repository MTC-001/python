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
list = ["張三","李四","趙名"]
for i in list:
    print(i)
for index,value in enumerate(list):
    print(index,value)
list.append("小明")
list.append("小麗")
list.extend(["趙造","小李","小王"])
print(list)
list.insert(0,"hhh")
print(list)