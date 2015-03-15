function [ce, ci] = to_col_sparmat( A )
%TO_COL_SPARMAT convert to coumn wise (my own) sparse matrix
% Input:
%  A: [L, ML] the sparse matrix
% Output:
%  ce: [K, ML] column wise elements
%  ci: [K, ML] column wise index

  % the maximum column #non-zero
  K = 0;
  ML = size(A,2);
  for i = 1 : ML
    nz = sum( A(:,i) > 0 );
    K = max(K, nz);
  end
 
  % make the sparse matrix
  ce = zeros([K,ML], 'like',A);
  ci = zeros([K,ML]);
  for i = 1 : ML
    ind = find( A(:,i) > 0 );
    nz  = numel(ind); % can < K !!!
    ce(1:nz, i) = A(ind,i);
    ci(1:nz, i) = ind(:);
  end
  
end

