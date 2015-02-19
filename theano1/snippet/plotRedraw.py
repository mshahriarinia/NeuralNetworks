import time
import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 1000, 0, 1])
plt.ion()
plt.show()

for i in range(1000):
    # ADDED TO CLEAR THE FIGURE
    plt.clf()
    plt.axis([0, 1000, 0, 1])
    plt.ion()
    # ADDED TO CLEAR THE FIGURE
    y = np.random.random()
    plt.scatter(i, y)
    plt.draw()
    time.sleep(0.005)
