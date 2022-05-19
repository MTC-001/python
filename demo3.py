import imp
import requests
import re

# url = 'https://fanyi.baidu.com/sug'
# s = input("要翻譯的單詞：")
# data = {
#     "kw" : s
# }
# resp = requests.post(url=url,data=data)
# print(resp.json())
# resp.close()

# 匹配所有符合正則的內容
# res = re.findall(r"\d+","電話號碼是11233,444545")
# print(res)
# for i in res:
#     print(i)
# 匹配所有符合正則的內容，返回迭代器
# it = re.finditer(r"\d+","電話號碼是11233,444545")
# for i in it:
#     print(i.group())
# 匹配一個符合正則的內容
res = re.search(r"\d+","手機號碼11111111,222222222,333333")
print(res.group())
# 從頭開始匹配一個符合正則的內容
res = re.match(r"\d+","12344,343")
print(res.group())