from matplotlib import pyplot as plt
file = open("output.txt", "r")
output = file.read().split()
time = [float(output[i]) * 1000 for i in range(9, len(output), 11)]
n = [i*i for i in range(2, 1024, 1)]
plt.plot(n, time)
plt.xlabel("Size of Matrix")
plt.ylabel("Runtime in miliseconds")
plt.show()
