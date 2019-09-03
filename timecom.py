
import scipy as sp

def dife(arr):
    for x in arr:
        if x != 'hondureno':
            pass
        else:
            pass

def igua(arr):
    for x in arr:
        if x == 'hondureno':
            pass
        else:
            pass

a = sp.array(['aleman', 'bulgaro', 'croata', 'danes', 'egipcio', 'fenicio',
              'guatemalteco', 'hondureno', 'indio'])

if __name__ == '__main__':
    from timeit import repeat
    print(repeat('igua(a)', setup='from __main__ import igua, dife, a', number=5000000))
