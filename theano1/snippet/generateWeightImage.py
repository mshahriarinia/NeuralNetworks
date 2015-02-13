from matplotlib import mpl,pyplot
import numpy as np

# make values from -5 to 5, for this example
zvals = np.random.rand(100,100)*10 - 5
#print(zvals)




# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['blue','black','red'])
bounds=[-6,-2,2,6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(zvals,interpolation='nearest',
                    cmap = cmap,norm=norm)

# make a color bar
#pyplot.colorbar(img,cmap=cmap,
 #               norm=norm,boundaries=bounds,ticks=[-5,0,5])

# gray scale image
fig = pyplot.figure(2)

zvals =  np.random.rand(28,28)

cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['black','white'],
                                           256)

img2 = pyplot.imshow(zvals,
                    interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')

pyplot.show()