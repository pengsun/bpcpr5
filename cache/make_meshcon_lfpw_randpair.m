%% mean shape
load('D:\data\facepose\lfpwnorm_matlab\tr_rescale_grad.mat', 'p');
% load('~/A/data/facepose/lfpwnorm_matlab/tr_rescale_grad.mat', 'p');

ppp = mean(p, 3);
%% pair wise distance
d = pdist(ppp', 'euclidean');
D = squareform(d);
%% connection with fixed radius circle
r = 0.20;
L = size(D,1);
Z = zeros(L, L);
for i = 1 : L
  ind = find( D(:,i) <= r );
  Z(ind, i) = 1; % includes itself
end
%% save
save( sprintf('lfpw_randpair_r%1.2f.mat',r), 'Z');
