class Accumulator(object):
    none = None
    def also(self, condition):
        self.none = not condition and (self.none is None or self.none)
        return condition
    pass


'''
A = Accumulator()
also = A.also


a = 1
b = 2

if also(a==0):
    print('0')

if also(a==1):
    print('1')
if also(a==2):
    print('2')
if also (a==1):
    print('3')
if also (a==2):
    print('4')
if also (a==2):
    print('5')
if A.none:  # else
    print("None of the conditions was met.")
'''
