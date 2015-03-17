%% mean shape
load('D:\data\facepose\lfpwnorm_matlab\tr_rescale_grad.mat', 'p');
% load('~/A/data/facepose/lfpwnorm_matlab/tr_rescale_grad.mat', 'p');

ppp = mean(p, 3);
%% pair wise distance
d = pdist(ppp', 'euclidean');
D = squareform(d);
%% kNN mask
knn = 2;
L = size(D,1);
Z = zeros(L, L);
for i = 1 : L
  [~,ind] = sort(D(:,i), 'ascend');
  Z(ind(1:knn), i) = 1; % includes itself
end
%% save
save( sprintf('lfpw_knn%d.mat',knn), 'Z');
