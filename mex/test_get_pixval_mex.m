%%
H = 192;
W = 192;
C = 3;
N = 128;
L = 68;
M = 5;

%% template
% load('lfpw_knn3.mat');
% assert( L == size(Z,1) );
% A = rand_pnts_knn_convcomb(Z, M);

load('lfpw_randpair_r0.20.mat');
assert( L == size(Z,1) );
A = rand_pnts_paircomb(Z, M);

[rcc, rci] = to_col_sparmat(A);
%% point
% load( 'D:\data\facepose\300-Wnorm_matlab\tr_rescale_grad.mat', 'p');
% % p = p(:,:, randsample(size(p,3), N) );
% p = p(:,:, 1:N );

load('pMean_300W.mat');
p = repmat(pMean, [1,1,N]);
%% to gpuArray
I = single( (1 : H*W*C) );
I = reshape(I, [H,W,C]);
I = repmat(I, [1,1,1,N]);
I = gpuArray( I );
p = gpuArray( single(p) );
rcc = gpuArray( single(rcc) );
rci = gpuArray( uint32(rci) );
%%
tic
[f,ind,pp] = get_pixval_mex(I,p,rcc,rci);
toc
% temp(I,p,rcc,rci);
% tt = gputimeit(@() ( get_pixval_mex(I,p,rcc,rci) ) )
%%
ff = gather(f);
ix = gather(ind);
%%
% ii = 125;
ii = randsample(N,1);

figure;
hold on;
plot( p(1,:,ii), p(2,:,ii), 'ro');
plot( pp(1,:,ii), pp(2,:,ii), 'b+');
set(gca,'ydir','reverse');
hold off;