import jieba
text = "我爱自然语言处理"
seg_list = jieba.cut(text, cut_all=True)
print(" ".join(seg_list))