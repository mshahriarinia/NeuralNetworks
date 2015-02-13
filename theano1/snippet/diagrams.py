
from matplotlib.mlab import normpdf
import numpy as nx
import pylab as p
from numpy import sin
import math

x = nx.arange(-4, 4, 0.1)

y=sin(x*math.pi)

p.plot(x,y, color='red', lw=2)

p.show()