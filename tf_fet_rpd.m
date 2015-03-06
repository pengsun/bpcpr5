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
      obj.pMean = pMean_;
      
      obj.is_bprop_in2 = false;
    end
    
    function [obj, dc] = fprop(obj, dc)   
      %%% in 
      p = dc{obj.in(1)}.a; % in 1: p [2,L,N]
      II = dc{obj.in(2)}.a; % in 2: II [W,H,3,N]
      I = squeeze( II(:,:,1,:) ); % I [W,H,N]
      
      %%% do it: generate the features
      if ( isempty(obj.d1) ) % initialize if necessary
        L = size(p,2);
        obj = init_param(obj, L);
      end
      % get the index to the random pixels
      obj.ind1 = get_yxind_posetform(size(I), p, obj.pMean, obj.d1);
      f1 = double( I(obj.ind1) ); % [MLN]
      obj.ind2 = get_yxind_posetform(size(I), p, obj.pMean, obj.d2);
      f2 = double( I(obj.ind2) ); % [MLN]
      % the values
      [~,L,N] = size(p);
      X = reshape(f1-f2, [obj.M, L, N]);
      
      %%% out 1: X [M, L, N]
      dc{obj.out(1)}.a = X;
    end
    
    function [obj, dc] = bprop(obj, dc)
      %%% out and in
      dX = dc{obj.out}.d; %  out .d: dX [M,L,N]
      p = dc{obj.in(1)}.a; % in 1.a: p [2, L, N]
      II = dc{obj.in(2)}.a; % in 2.a : II [W,H,3,N]
      %I  = squeeze( II(:,:,1,:) ); % [W,H,N]
      Gx = squeeze( II(:,:,2,:) ); % [W,H,N]
      Gy = squeeze( II(:,:,3,:) ); % [W,H,N]
      
      %%% bprop for p: in1.d 
      [~,L,N] = size(p);
      %obj.ind1 = get_yxind_posetform(size(I), p, obj.pMean, obj.d1); % [MLN]
      f1x = double( Gx(obj.ind1) ); % [MLN]
      f1x = reshape(f1x, [1, obj.M,L,N]); % [1, M,L,N]
      f1y = double( Gy(obj.ind1) ); % [MLN]
      f1y = reshape(f1y, [1, obj.M,L,N]); % [1, M,L,N]
      GG1 = cat(1, f1x,f1y); % [2,M,L,N]
      %obj.ind2 = get_yxind_posetform(size(I), p, obj.pMean, obj.d2); % [MLN]
      f2x = double( Gx(obj.ind2) ); % [MLN]
      f2x = reshape(f2x, [1, obj.M,L,N]); % [1, M,L,N]
      f2y = double( Gy(obj.ind2) ); % [MLN]
      f2y = reshape(f2y, [1, obj.M,L,N]); % [1, M,L,N]
      GG2 = cat(1, f2x,f2y); % [2,M,L,N]
      % delta
      dXdX = reshape(dX,[1,obj.M,L,N]); % [1,M,L,N]
      dXdX = cat(1, dXdX,dXdX); % [2,M,L,N]
      % times
      tmp = (GG1-GG2) .* dXdX; % [2,M,L,N]
      % in 1.d: dp [2,L,N]
      dc{obj.in(1)}.d = squeeze( sum(tmp,2) ); % squeeze( [2,1,L,N] )
       
      %%% whether bprop for I? (typically doesn't need it when training)
      dc{obj.in(2)}.d = zeros( size( dc{obj.in(2)}.a ) ); % [W,H,3,N]
      if (~obj.is_bprop_in2), return; end
      
      %%% bprop for I: in2.d
      tmp1 = zeros( size(Gx) );   % [W,H,N]
      tmp1(obj.ind1) = dX(:);     % [W,H,N], with MLN non-zero elements
      tmp2 = zeros( size(tmp1) ); % [W,H,N]
      tmp2(obj.ind2) = dX(:);     % [W,H,N], with MLN non-zero elements
      tmp = tmp1 - tmp2;          % [W,H,N]
      % write it
      dc{obj.in(2)}.d(:,:,1,:) = tmp; % leave the other 2 channels
    end % bprop
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
