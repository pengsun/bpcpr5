function pAll = get_pAll_posetform(p, pMean, d)
%GET_PALL_POSETFORM get pAll = (pMean+ Rotate_Scale(d)) at coordinates 
%defined by p
% p: [2,L,N]
% pMean: [2,L]
% d: [2,M,L] 
% pAll: [2,M,L,N] 

  %
  [~,M,L] = size(d);
  [~,~,N] = size(p);
  pAll = zeros(2,M,L,N); % [2,M,L,N] 
  % for each image
  for i = 1 : N
    % similarity transform: from pMean to p
    z = cp2tform(pMean', p(:,:,i)', 'nonreflective similarity');
    % get dd = ratate_scale(d) % [2,M,L]
    % only ratation and scaling are needed
    dd = z.tdata.T(1:2,1:2) * reshape(d, [2,M*L]);
    dd = reshape(dd, [2,M,L]); % [2,M,L]
    % the centroids
    pp = p(:,:,i); % [2,L]
    pp = reshape(pp, [2,1,L]); % [2,1,L]
    pp = repmat(pp, [1,M,1]); % [2,M,L]
    % feature points
    pFet = pp + dd; % [2,M,L]  
    pAll(:,:,:, i) = pFet; % [2,M,L]  
  end % for i
  
end % get_pAll_posetform