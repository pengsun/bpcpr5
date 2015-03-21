classdef batpose_gentor
  %BATPOSE_GENTOR Summary of this class goes here
  %   Detailed explanation goes here

  properties
    N;       % number of original #instances
    i_bat;   % current batch count
    num_bat; % 
  end
  
  properties
    h1;
    h2;
  end
  
  methods
    function ob = batpose_gentor()
      ob = reset(ob, 888, 16888, 168); % Why the numbers?? -_-
    end
    
    function ob = reset(ob, N, Nstar, bat_sz)
      ob.h1 = bat_gentor();
      ob.h1 = reset(ob.h1, Nstar, bat_sz);
      ob.h2 = bat_gentor();
      ob.h2 = reset(ob.h2, Nstar, bat_sz);
      
      ob.N = N;
      ob.i_bat = 1;
      assert( ob.h1.num_bat == ob.h2.num_bat );
      ob.num_bat = ob.h1.num_bat;
    end % reset
    
    function [idx1,idx2] = get_idx (ob, ib)
      % the index in {1,...,Nstar}
      idx1 = get_idx(ob.h1, ib);
      idx2 = get_idx(ob.h2, ib);
      
      % the index in {1,...,N}
      idx1 = 1 + mod(idx1-1, ob.N);
      idx2 = 1 + mod(idx2-1, ob.N);
    end % get_idx
    
    function [idx1,idx2] = get_idx_orig (ob, ib)
      % the index in {1,...,Nstar}
      idx1 = get_idx_orig(ob.h1, ib);
      idx2 = get_idx_orig(ob.h2, ib);
      
      % the index in {1,...,N}
      idx1 = 1 + mod(idx1-1, ob.N);
      idx2 = 1 + mod(idx2-1, ob.N);
    end % get_idx_orig    
   
    function [bat_I,bat_pGT,bat_pInit] = get_data (ob, I, pGT, ib)
      [ix1,ix2] = get_idx(ob, ib);
      bat_pGT   = pGT(:,:,ix1);
      bat_I     = I(:,:,:,ix1);
      
      % transform the mean shape as the initial pose
      bat_pInit = zeros( size(bat_pGT) );
      pMean = mean(pGT,3); % [2,L]
      parfor i = 1 : numel(ix2)
        ix = ix2(i);
        
        p_moving = pGT(:,:,ix); % [2,L]
        z = fitgeotrans(p_moving', pMean', 'nonreflectivesimilarity');
        p_moved = transformPointsForward(...
          z, [p_moving(1,:)', p_moving(2,:)'] );
        
        bat_pInit(:,:,i) = p_moved';
      end
    end % get_data
    
    function [bat_I,bat_pGT,bat_pInit] = get_data_orig (ob, I, pGT, ib)
      [ix1,ix2] = get_idx_orig(ob, ib);
      bat_pGT   = pGT(:,:,ix1);
      bat_I     = I(:,:,:,ix1);
      
      % transform the mean shape as the initial pose
      bat_pInit = zeros( size(bat_pGT) );
      pMean = mean(pGT,3); % [2,L]
      for i = 1 : numel(ix2)
        ix = ix2(i);
        
        p_moving = pGT(:,:,ix); % [2,L]
        z = cp2tform(p_moving', pMean', 'nonreflective similarity');
        p_moved = tformfwd(z, p_moving(1,:)', p_moving(2,:)');
        
        bat_pInit(:,:,i) = p_moved';
      end
    end % get_data_orig
    
  end % methods
  
end % batpose_gentor

