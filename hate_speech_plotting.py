from matplotlib import pyplot as plt
import numpy as np


array = np.genfromtxt("./preprocessed_data/final_hate_num.csv", delimiter=",")[1:5, 1:]
print(array.shape)

fig = plt.figure()
color = ['green', 'red', "black", "yellow", "blue"]

x = [i for i in range(1, 21)]

for i in range(array.shape[0]):
    plt.plot(x, array[i], color=color[i])

fig.show()