class person():
    
    # 構造函數
    def __init__(self,name,age):
        self.name = name
        self.age = age
    
    # 析構函數
    def __del__(self):
        print("對象銷毀了")
    
    def sleep(self):
        print("睡覺了")
        
    def eat(self):
        print("吃飯了")
class girl(person):
    
    def __init__(self,name,age,weight):
        # person.__init__(self,name,age)
        super(girl,self).__init__(name,age)
        self.weight = weight
    
    def say(self):
        print("我是女生")   