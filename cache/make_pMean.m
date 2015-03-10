%% 300W
fn_data = fullfile(...
  'D:\data\facepose\300-Wnorm_matlab',... % directory 
  'tr_rescale_grad.mat');                 % file name
load(fn_data, 'p');
pMean = mean(p, 3);
save('pMean_300W.mat', 'pMean');