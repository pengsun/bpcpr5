function pnts = rand_pnts_unit_circle(N)
%RAND_PNTS_UNIT_CIRCLE Summary of this function goes here
%   Detailed explanation goes here
  
  % rejection sampling
  NN = round( 1.5*N );
  pnts = zeros(2,0);
  
  % rejection sampling until desired #points
  while( true )
    tmp = 2*( rand(2,NN) - 0.5 );
    zz = sum(tmp.^2, 1); % radius 
    pnts = [pnts, tmp(:, zz<1)]; % appden to tail those eligible points
    NN = size(pnts,2);
    if (NN>=N), break; end
  end
  
  pnts = pnts(:, 1:N);

end

