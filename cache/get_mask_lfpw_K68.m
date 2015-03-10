function A = get_mask_lfpw_K68(M, m, knn)
%GET_MASK_LFPW Summary of this function goes here
% M: [1], typically 5
% m: [1], typically 6
% knn: [1], #nearest neighbors
% A: [ML, mL], L = 68
  
  % the knn indicator
  tmp = load( sprintf('lfpw_knn%d.mat',knn) );
  Z = tmp.Z; clear tmp;
  % [L, L], where L = 68, each column is 0/1 mask  
  
  % kroneck product
  A = kron(Z, ones(M,m));
  
end

