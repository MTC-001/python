import os
from tkinter import *
from urllib.request import urlretrieve
import requests
import jsonpath




def song_load(title,author,url):
    zi_file_path="G:\music"
    if not os.path.exists(zi_file_path):
        os.makedirs("music",exist_ok=True) #exist_ok=True 如果当前目录下有相同名称的文件夹则覆盖
    path="music\{}.mp3".format(title)
    listbox.insert(END,"歌曲：{}正在下载.....".format(title))
    listbox.insert(END,"当前下载路径{}".format(os.getcwd()))
    listbox.insert(END,os.getcwd())
    listbox.see(END)
    listbox.update()
    path1=os.getcwd()+"\\music\{}.mp3".format(title)
    print(path1)
    if not os.path.exists(path1):
        urlretrieve(url,path)
        listbox.insert(END, "歌曲：{}下载完毕，请试听".format(title))
    else:
        listbox.insert(END,"{}已存在!".format(title))

    listbox.see(END)
    listbox.update()

def get_music():
    name=entry.get()
    url='https://music.liuzhijin.cn/'
    params={
        'input': name,
        'filter': 'name',
        'type': 'qq',
        'page': 1,
    }
    headers={
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'}
    res=requests.post(url=url,headers=headers,data=params)
    data=res.json()
    title=jsonpath.jsonpath(data,"$..title")[0]
    author=jsonpath.jsonpath(data,"$..author")[0]
    url=jsonpath.jsonpath(data,"$..url")[0]
    print(title)
    print(author)
    print(url)
    song_load(title,author,url)


# 创建窗口
# 1、创建画布
root = Tk()
# 2设置title
root.title("音乐下载器")
# 3、设置窗口大小
root.geometry('980x680')
# 4、创建标签控件提示用户
label = Label(root, text="请输入需要下载的歌曲：", font=("隶书", 27))
# 5、设置标签控件的位置
label.grid()
# 6、创建输入框
entry = Entry(root, font=("隶书", 39))
entry.grid(row=0, column=1)
listbox = Listbox(root, font=("隶书", 28), width=50, heigh=15)
listbox.grid(row=1, columnspan=2)
button1 = Button(root, text="开始下载", font=("宋体", 15), command=get_music)  # command=方法 绑定按钮需要实现的功能代码
button1.grid(row=2, column=0, sticky=W)
button2 = Button(root, text="退出程序", font=("宋体", 15), command=root.quit)
button2.grid(row=2, column=1, sticky=E)
root.mainloop()
