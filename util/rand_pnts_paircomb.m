function A = rand_pnts_paircomb(Z, M)
%RAND_PNTS_CONVCOMB pair Combinations using the mesh connection
% Input:
%   Z: [L, L] 0/1 mesh connection (a mask/template)
%   M: [1] #points generated per point/landmark
% Output:
%   A: [L, ML] convex combination coefficients, A(:,i) is a sum-to-one vec
%

  L = size(Z,1); 
  assert(L == size(Z,2));
  A = zeros(L, M*L);
  
  % for each point, generate M points around it
  for i = 1 : L
    % #points for the convex hull
    ind = find( Z(:,i) > 0 );
    KK = numel(ind);
    assert(KK >= 2, 'KK >= 2: connect to at least one point.');
    K = 2;
    
    for j = 1 : M
      % generate the sum-to-one coefficients with #points components
      co = rand(1,K);
      co = co./sum(co);
      % assign it to the right column of A
      ii = (i-1)*M + j;
      indpair = ind( randsample(KK, K) );
      assert( numel(indpair) == K );
      A(indpair, ii) = co(:);
    end % for j
  end % for i

end

