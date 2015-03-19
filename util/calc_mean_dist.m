function md = calc_mean_dist(p1, p2)
%CALC_MEAN_DIST Summary of this function goes here
%   Detailed explanation goes here
  
  tmp = p1 - p2; % [2,L,N]
  tmp = tmp.^2; % [2,L,N]
  tmp = sqrt( sum(tmp, 1) ); % [1,L,N]
  tmp = squeeze( tmp ); % [L,N]
  
  md = mean( tmp(:) );
end

