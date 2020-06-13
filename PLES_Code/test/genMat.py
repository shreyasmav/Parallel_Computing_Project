from random import randint,seed
from sys import argv
n = int(argv[1])
seed(n)
print(n)
for i in range(n):
	for j in range(n):
		print(randint(0, n), end=' ')
	print()
