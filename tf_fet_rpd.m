classdef tf_fet_rpd < tf_i
  %TF_RPD Random Pixel Difference, using similarity tform
  %   Detailed explanation goes here
  
  properties
    r; % [1] radius
    M; % [1] #pixel difference pairs per point
    
    d1; % [2, M, L] random points in canonical coordinate (<=r)
    d2;    
    
    pMean; % [2,L]
%     pFet1; % [2,M,L] for feature 1
%     pFet2; % [2,M,L]
    
    ind1; % index
    ind2; 
    
    is_bprop_in2; % true: bprop for in 2 (the image I); false: don't
  end
  
  methods
    function obj = tf_fet_rpd(pMean_)
      obj.r = 0.1;
      obj.M = 2;
      obj.pMean = single( pMean_ ); 
      
      obj.is_bprop_in2 = false;
      
      %%% input output
      obj.i = [n_data(),n_data()];
      obj.o = n_data();
    end
    
    function ob = fprop(ob)
      ttt = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%% in 
      p = ob.i(1).a; % in 1: p [2L,N]
      % enforced cpu array !!!
      p = single( gather(p) );
      II = ob.i(2).a; % in 2: II [W,H,3,N]
      I = squeeze( II(:,:,1,:) ); % I [W,H,N]
      
      %%% do it: generate the features
      if ( isempty(ob.d1) ) % initialize if necessary
        L = size(p,2);
        ob = init_param(ob, L);
      end
      % get the index to the random pixels
      pm = gather( ob.pMean ); % enforced cpu array
      ob.ind1 = get_yxind_posetform(size(I), p, pm, ob.d1);
      f1 = I(ob.ind1) ; % [MLN]
      ob.ind2 = get_yxind_posetform(size(I), p, pm, ob.d2);
      f2 = I(ob.ind2) ; % [MLN]
      % the values
      [~,L,N] = size(p);
      X = reshape(f1-f2, [ob.M, L, N]);
      
      %%% out 1: X [M, L, 1, N]
      % [M, L, N] -> [M,L,1,N], the matconvnet format
      ob.o.a = reshape(X, [ob.M,L,1,N]);
      ttt = toc(ttt); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %fprintf(' tf_rpd.fprop: %.4fs ', ttt);
    end % fprop
    
    function ob = bprop(ob)
      ttt = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%% out and in
      dX = ob.o.d;      % out .d: [M,L,1,N]
      dX = squeeze(dX); % [M,L,N]
      p  = ob.i(1).a; % in 1.a: p [2, L, N]
      II = ob.i(2).a; % in 2.a: II [W,H,3,N]
      %I  = squeeze( II(:,:,1,:) ); % [W,H,N]
      Gx = squeeze( II(:,:,2,:) ); % [W,H,N]
      Gy = squeeze( II(:,:,3,:) ); % [W,H,N]
      
      %%% bprop for p: in1.d 
      [~,L,N] = size(p);
      %obj.ind1 = get_yxind_posetform(size(I), p, obj.pMean, obj.d1); % [MLN]
      f1x = double( Gx(ob.ind1) ); % [MLN]
      f1x = reshape(f1x, [1, ob.M,L,N]); % [1, M,L,N]
      f1y = double( Gy(ob.ind1) ); % [MLN]
      f1y = reshape(f1y, [1, ob.M,L,N]); % [1, M,L,N]
      GG1 = cat(1, f1x,f1y); % [2,M,L,N]
      %obj.ind2 = get_yxind_posetform(size(I), p, obj.pMean, obj.d2); % [MLN]
      f2x = double( Gx(ob.ind2) ); % [MLN]
      f2x = reshape(f2x, [1, ob.M,L,N]); % [1, M,L,N]
      f2y = double( Gy(ob.ind2) ); % [MLN]
      f2y = reshape(f2y, [1, ob.M,L,N]); % [1, M,L,N]
      GG2 = cat(1, f2x,f2y); % [2,M,L,N]
      % delta
      dXdX = reshape(dX,[1,ob.M,L,N]); % [1,M,L,N]
      dXdX = cat(1, dXdX,dXdX); % [2,M,L,N]
      % times
      tmp = (GG1-GG2) .* dXdX; % [2,M,L,N]
      % in 1.d: dp [2,L,N]
      ob.i(1).d = squeeze( sum(tmp,2) ); % squeeze( [2,1,L,N] )
       
      ttt = toc(ttt); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %fprintf(' tf_rpd.bprop: %.4fs ', ttt);
      %%% whether bprop for I? (typically doesn't need it when training)
      ob.i(2).d = zeros( size(ob.i(2).a) ); % [W,H,3,N]
      if (~ob.is_bprop_in2), return; end
      
      %%% bprop for I: in2.d
      tmp1 = zeros( size(Gx) );   % [W,H,N]
      tmp1(ob.ind1) = dX(:);     % [W,H,N], with MLN non-zero elements
      tmp2 = zeros( size(tmp1) ); % [W,H,N]
      tmp2(ob.ind2) = dX(:);     % [W,H,N], with MLN non-zero elements
      tmp = tmp1 - tmp2;          % [W,H,N]
      % write it
      ob.i(2).d(:,:,1,:) = tmp; % leave the other 2 channels

    end % bprop
    
    function ob = cvt_data(ob)
      % convert internal state
      ob.pMean = ob.ab.cvt_data( ob.pMean );
      % convert other
      ob = cvt_data@tf_i(ob);
    end % cvt_data
    
  end % methods
  
  %%% helpers
  methods
    function obj = init_param(obj, L)
    % L: [1] #points
      
      % set the random difference coordinates
      obj.d1 = rand_pnts_unit_circle(obj.M * L) * obj.r;
      obj.d1 = reshape(obj.d1, [2,obj.M,L]);

      obj.d2 = rand_pnts_unit_circle(obj.M * L) * obj.r;
      obj.d2 = reshape(obj.d2, [2,obj.M,L]);
      
%       tmp_pMean = reshape(obj.pMean, [2,1,L]); % [2,1,L]
%       tmp_pMean = repmat(tmp_pMean, [1,obj.M,1]); % [2,M,L]
%       obj.pFet1 = tmp_pMean + obj.d1; % [2,M,L]
%       obj.pFet2 = tmp_pMean + obj.d2; % [2,M,L]
    end
  end
  
end
