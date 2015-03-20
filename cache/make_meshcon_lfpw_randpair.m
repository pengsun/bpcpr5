%% mean shape
load('D:\data\facepose\lfpwnorm_matlab\tr_rescale_grad.mat', 'p');
% load('~/A/data/facepose/lfpwnorm_matlab/tr_rescale_grad.mat', 'p');

ppp = mean(p, 3);
%% pair wise distance
d = pdist(ppp', 'euclidean');
D = squareform(d);
%% connection with fixed radius circle
r = 0.09;
L = size(D,1);
Z = zeros(L, L);
for i = 1 : L
  ind = find( D(:,i) <= r );
  assert( numel(ind) >=1 );
  
  % ensure it connects to at least one point: numel(ind) >= 2
  if (numel(ind) == 1)
    [~,ind_sort] = sort(D(:,i), 'ascend');
    ind(end+1) = ind_sort(2); % ind_sort(1): itself; ind_sort(2): nearest negighbor
  end
  
  Z(ind, i) = 1; % includes itself
end
%% save
save( sprintf('lfpw_randpair_r%1.2f.mat',r), 'Z');
