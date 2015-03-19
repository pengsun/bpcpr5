function [nd, ndnd] = calc_pupil_dist(p, pGT)
%CALC_PUPIL_DIST Summary of this function goes here
%   Detailed explanation goes here
  
  for i = 1 : size(p, 3)
    % mean distance
    md(i) = calc_mean_dist( p(:,:,i), pGT(:,:,i) );
    
    % distance between pupils
    pp1 = pGT(:, 37:42, i);
    pp2 = pGT(:, 43:48, i);
    pd(i) = calc_mean_dist(pp1, pp2);
    
    %
    ndnd(i) = md(i) / pd(i);
  end
 
  nd = mean(ndnd);
end

