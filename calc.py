import numpy as np

def calc(a):
	b = 0
	print(a)
	for i in range(len(a)):
		b = a[i]+b
	return b

print(calc(np.random.randint(60,size=8)))
