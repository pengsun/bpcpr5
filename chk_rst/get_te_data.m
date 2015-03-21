function [pInit,I,pGT] = get_te_data(I,pGT, ind)
% pInit
load('pMean_300W.mat', 'pMean');
N = size(pGT,3);
pInit = repmat(pMean, 1,1,N);
% make them (to avoid the bug when there is only one instance)
I     = cat(4, I(:,:,:,ind), I(:,:,:,ind) );
pGT   = cat(3, pGT(:,:, ind), pGT(:,:, ind) );
pInit = cat(3, pInit(:,:,ind),pInit(:,:,ind) );
