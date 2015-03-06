function ind = get_yxind_posetform(szI, p, pMean, d)
% pixel index list to the image
% szI: [H,W,N]
% p: [2,L,N]
% pMean: [2,L]
% d: [2,M,L] 
% ind: [MLN] 
  
  [~,L,N] = size(p);
  [~,M,~] = size(d);
  
  % coordinates for all the [M L N] points
  pAll = get_pAll_posetform(p, pMean, d);

  % to image coordinate (pixels)
  pAll(1,:,:,:) = round( szI(2) * pAll(1,:,:,:) );
  pAll(2,:,:,:) = round( szI(1) * pAll(2,:,:,:) );
  
  % x (dim2) [M,L,N]
  ix = pAll(1,:,:,:);
  % clap it
  W = szI(2);
  ix( ix<1 ) = 1; ix( ix>W ) = W;
  %  y (dim1) [M,L,N]
  iy = pAll(2,:,:,:);
  % clap it
  H = szI(1);
  iy( iy<1 ) = 1; iy( iy>H ) = H;
  
  % get the linear index
  imgId = zeros(1,1,N);
  imgId(1,1,:) = (1:N);
  imgId = repmat(imgId, [M,L,1]);
  ind = sub2ind(szI, iy(:), ix(:), imgId(:));

end