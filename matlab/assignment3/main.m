% I use this for my experiments and illustrations

  
a3(1e7, 7, 10, 0, 0, false, 4)

  %% Visualize data
  
from_data_file = load('data.mat');
data = from_data_file.data;
  
for i=1:50
    idx = i*10+4; % idx = i; % for all the numbers
figure, imshow(reshape(datas.training.inputs(:,idx), [16,16])');
[1 2 3 4 5 6 7 8 9 0] * datas.training.targets(:,idx)

end