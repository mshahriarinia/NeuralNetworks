--[[
To run:
    $ qlua image.lua

for using qtlua, start torch in your terminal with:
    $ qlua
instead of
    $ th

 you can also run like this:
    $ qlua -e "require('trepl')()"    
    th>  require('image.lua')

--]]

require 'nn'
require 'image'

i = image.lena()

image.display(i)

image.display{image={image.lena(), image.lena()},gui=false}

