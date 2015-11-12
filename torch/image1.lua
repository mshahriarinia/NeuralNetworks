--[[
To run:
    $ qlua image.lua

For using qtlua, start torch in your terminal with:
    $ qlua
instead of
    $ th

Or, you can also run like this:
    $ qlua -e "require('trepl')()"    
    th>  require('image.lua')

Or, from within qlua
    $ qlua
    > dofile 'getstarted.lua'

--]]

require('nn')
require('image')

i = image.lena()

image.display(i)

image.display{image={image.lena(), image.lena()},gui=false}

--  visualizing internal states, and convolution filters:
n = nn.SpatialConvolution(1,64,16,16)
image.display(n.weight)

n = nn.SpatialConvolution(1,16,12,12)
res = n:forward(image.rgb2y(image.lena()))
image.display(res:view(16,1,501,501))
