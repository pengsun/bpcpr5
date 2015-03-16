%%
H = 192;
W = 192;
C = 3;
N = 128;
L = 68;
M = 5;
K = 2;

% template
load('lfpw_knn8.mat');
assert( L == size(Z,1) );
A = rand_pnts_knn_convcomb(Z, M);
[rcc, rci] = to_col_sparmat(A);
% point
load( 'D:\data\facepose\300-Wnorm_matlab\tr_rescale_grad.mat', 'p');
% p = p(:,:, randsample(size(p,3), N) );
p = p(:,:, 1:N );
%%
I = gpuArray( zeros([H,W,C,N], 'single') );
p = gpuArray( single(p) );
rcc = gpuArray( single(rcc) );
rci = gpuArray( uint32(rci) );
%%
[f,ind,pp] = get_pixval_mex(I,p,rcc,rci);
% temp(I,p,rcc,rci);
% tt = gputimeit(@() ( get_pixval_mex(I,p,rcc,rci) ) )
%%
figure;

ii = 27;
hold on;
plot( p(1,:,ii), p(2,:,ii), 'ro');
plot( pp(1,:,ii), pp(2,:,ii), 'b+');
set(gca,'ydir','reverse');
hold off