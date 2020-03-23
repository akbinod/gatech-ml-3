import numpy as np

# a = np.arange(start = 0, stop = 5, step = 1)
# print(a)

a=[1,2,3,4]
a = [i * - 1 for i in a]
aa = {}
aa['sa'] = 1
aa['rhc'] = 2
for key in aa:
	print(key)

for index, b in enumerate(a):
	print(index,b)

for idx, key in enumerate(aa):
	print(idx, key, aa[key])
