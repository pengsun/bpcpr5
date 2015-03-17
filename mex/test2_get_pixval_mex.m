%%
H = 43;
W = 89;
C = 3;
N = 101;

L = 68;
M = 5;
%% the image
I = single( (1 : H*W*C) );
I = reshape(I, [H,W,C]);
I = repmat(I, [1,1,1,N]);
%% the template rcc, rci
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
%% test1
for n = 1 : size(pp,3)
  cmp = gather(pp(:,:,1) == pp(:,:,n)) ;
  assert( all(cmp(:))==true, 'fail at n = %d', n);
end
%% test2
for n = 1 : N
  ML = M*L;
  
  ix1 = ind(1:ML);
  If1 = I(ix1);
  
  ixn = ind( (n-1)*ML + (1:ML) );
  Ifn = I(ixn);
  
  cmp = gather(If1==Ifn);
  assert( all(cmp(:))==true, 'fail at n = %d', n);

end
%% test3
for n = 1 : N
  ML = M*L;

  ixn = ind( (n-1)*ML + (1:ML) );
  Ifn = I(ixn);
  fn  = f( (n-1)*ML + (1:ML) );
  
  cmp = gather(Ifn==fn);
  assert( all(cmp(:))==true, 'fail at n = %d', n);

end
%% to cpu array
ff = gather(f);
ix = gather(ind);
%% plot 
% ii = 125;
ii = randsample(N,1);

figure;
hold on;
plot( p(1,:,ii), p(2,:,ii), 'ro');
plot( pp(1,:,ii), pp(2,:,ii), 'b+');
set(gca,'ydir','reverse');
hold off;