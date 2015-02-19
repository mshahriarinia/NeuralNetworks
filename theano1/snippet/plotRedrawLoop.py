import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.ion()
plt.show(block=True)

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', 
                                           ['black','white'],
                                           256)

for i in range(10):
    # ADDED TO CLEAR THE FIGURE
    plt.clf()
    plt.ion()
    # ADDED TO CLEAR THE FIGURE
    
    zvals =  np.random.rand(28,28)
    img2 = plt.imshow(zvals,
                    interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')
    
    
    plt.draw()
    time.sleep(0.005)

print "end"

raw_input("Press Enter to close...")