% I use this for my experiments and illustrations
from_data_file = load('data.mat');
  datas = from_data_file.data;
  
for i=1:50
    idx = i*10+4; % idx = i; % for all the numbers
figure, imshow(reshape(datas.training.inputs(:,idx), [16,16])');
[1 2 3 4 5 6 7 8 9 0] * datas.training.targets(:,idx)

end