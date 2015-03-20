classdef tf_fet_rpdni_mex < tf_i
  %TF_RPDNI Random Pixel Difference, neighborhood indexing, mex impl
  %   Detailed explanation goes here
  
  properties
    M; % [1] #pixel difference pairs per point
    
    % random combination that mimics the sparse matrix with up to K non
    % zeros elements
    rcc1; % [K, ML] random combination coefficients
    rci1; % [K, ML] point index  
    rcc2;
    rci2;
    
    Z; % [L,L] 0/1 template for knn, Z(:,i) indicates the knn for point i
    
    ind1; % [MLN] the feature's linear index to the image 
    ind2; 
    
    is_bprop_in2; % true: bprop for in 2 (the image I); false: don't
  end
  
  methods
    function obj = tf_fet_rpdni_mex(Z)
      %%% internal data
      obj.M = 2;
      obj.Z = Z; 
      
      obj.is_bprop_in2 = false;
      
      %%% input output
      obj.i = [n_data(),n_data()];
      obj.o = n_data();
    end
    
    function ob = fprop(ob)
      %ttt = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%% in
      ob.i(1).a( ob.i(1).a<0.0 ) = 0.0;
      ob.i(1).a( ob.i(1).a>1-eps ) = 1-eps;
      p = ob.i(1).a; % in 1: p [2,L,N]
      I = ob.i(2).a; % in 2: II [W,H,3,N]
      
      
      %%% do it: generate the features
      if ( isempty(ob.rcc1) ) % initialize if necessary
        ob = init_param(ob);
      end
      % the first %%%% TODO: the right implementation!
      [f1, ob.ind1] = get_pixval_mex(I, p, ob.rcc1, ob.rci1); % [MLN]
      % the second %%%% TODO: the right conversion!
      [f2, ob.ind2] = get_pixval_mex(I, p, ob.rcc2, ob.rci2); % [MLN]

      %%% out 1: X [M, L, 1, N]
      % the values: [M*L*N] -> [M,L,1,N], the matconvnet format
      [~,L,N] = size(p);
      ob.o.a = reshape(f1-f2, [ob.M, L, 1, N]);
      ob.ab.sync();
      %ttt = toc(ttt); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %fprintf(' tf_rpd.fprop: %.4fs ', ttt);
    end % fprop
    
    function ob = bprop(ob)
      %ttt = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%% out and in
      dX = ob.o.d;      % [M,L,1,N]
      dX = squeeze(dX); % [M,L,N]
      p  = ob.i(1).a;   % [2, L, N]
      II = ob.i(2).a;   % [W,H,3,N]
      WH = size(II,1) * size(II,2);
      
      %%% bprop for p: in1.d 
      [~,L,N] = size(p);
      % 
      f1x = II( ob.ind1 + WH ); % index to [:,:,2,:] % [MLN]
      f1x = reshape(f1x, [1, ob.M,L,N]); % [1, M,L,N]
      f1y = II( ob.ind1 + 2*WH ); % index to [:,:,3,:] % [MLN]
      f1y = reshape(f1y, [1, ob.M,L,N]); % [1, M,L,N]
      GG1 = cat(1, f1x,f1y); % [2,M,L,N]
      % 
      f2x = II( ob.ind2 + WH ); % [MLN]
      f2x = reshape(f2x, [1, ob.M,L,N]); % [1, M,L,N]
      f2y = II( ob.ind2 + 2*WH ); % [MLN]
      f2y = reshape(f2y, [1, ob.M,L,N]); % [1, M,L,N]
      GG2 = cat(1, f2x,f2y); % [2,M,L,N]
      % delta
      dXdX = reshape(dX,[1,ob.M,L,N]); % [1,M,L,N]
      dXdX = cat(1, dXdX,dXdX); % [2,M,L,N]
      % times
      tmp = (GG1-GG2) .* dXdX; % [2,M,L,N]
      %%% in 1.d: dp [2,L,N]
      ob.i(1).d = squeeze( sum(tmp,2) ); % [2,L,N] = squeeze( [2,1,L,N] )
      ob.ab.sync();
      %ttt = toc(ttt); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %fprintf(' tf_rpd.bprop: %.4fs ', ttt);
      
      %%% whether bprop for I? (typically doesn't need it when training)
      ob.i(2).d = zeros( size(ob.i(2).a) ); % [W,H,3,N]
      if (~ob.is_bprop_in2), return; end
      
      %%% bprop for I: in2.d
      tmp1 = zeros( size(II) ); % [W,H,3,N]
      tmp1( ob.ind1 ) = dX(:);  % [W,H,3,N], with MLN non-zero elements
      tmp2 = zeros( size(II) ); % [W,H,3,N]
      tmp2( ob.ind2 ) = dX(:);  % [W,H,3,N], with MLN non-zero elements
      tmp = tmp1 - tmp2;        % [W,H,3,N]
      % write it
      ob.i(2).d(:,:,1,:) = tmp; % leave the other 2 channels
      ob.ab.sync();
    end % bprop
    
    function ob = cvt_data(ob)
      % convert internal state
      ob.rcc1 = ob.ab.cvt_data( ob.rcc1 );
      ob.rci1 = ob.ab.cvt_data( ob.rci1 );
      ob.rcc2 = ob.ab.cvt_data( ob.rcc2 );
      ob.rci2 = ob.ab.cvt_data( ob.rci2 );
      
      ob.rci1 = uint32( ob.rci1 );
      ob.rci2 = uint32( ob.rci2 );
      % convert other
      ob = cvt_data@tf_i(ob);
    end % cvt_data
    
  end % methods
  
  %%% helpers
  methods
    function ob = init_param(ob)
      A1 = rand_pnts_paircomb(ob.Z, ob.M);
      [ob.rcc1, ob.rci1] = to_col_sparmat(A1);
      A2 = rand_pnts_paircomb(ob.Z, ob.M);
      [ob.rcc2, ob.rci2] = to_col_sparmat(A2);
      
      % convert to the right data format
      ob.rcc1 = ob.ab.cvt_data( ob.rcc1 );
      ob.rci1 = ob.ab.cvt_data( ob.rci1 );
      ob.rcc2 = ob.ab.cvt_data( ob.rcc2 );
      ob.rci2 = ob.ab.cvt_data( ob.rci2 );
      
      ob.rci1 = uint32( ob.rci1 );
      ob.rci2 = uint32( ob.rci2 );
    end % init_param
  end % methods
  
end % tf_fet_rpdni
