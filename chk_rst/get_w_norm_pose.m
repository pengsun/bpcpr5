function nm = get_w_norm_pose( w )
%GET_W_MAXNORM Summary of this function goes here
%  Input:
%    w: [a,b,c,d]
%  Output:
%    nm: [d]
%

  assert( ndims(w) == 4 );
  [a,b,c,d] = size(w);
  assert(c==1);
  nm = ones(1,d);
  
  w = squeeze(w);
  for i = 1 : d
    tmp = w(:,:,i);
    nm(i) = norm( tmp(:) );
  end % i
  
end % get_w_maxnorm

