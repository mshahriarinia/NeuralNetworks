
import numpy as nx
from numpy import sin
import pylab as p
import math

x = nx.arange(-4, 4, 0.1)

y=sin(x*math.pi)

p.plot(x,y, color='red', lw=2)

p.show()