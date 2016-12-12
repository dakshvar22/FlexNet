__author__ = 'daksh'


class hello():
    def __init__(self,name):
        self.varCheck = name

def set(l1,name):
    l1.varCheck = name

h = hello('original')
print h.varCheck
set(h,'changed')
print h.varCheck