classdef tfw_linloclin < tfw_i
  %tfw_gpu_linloclin A regressor: local linear + Relu + Dropout + linear
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = tfw_linloclin(mask_, sz2)
      %%% set the connection structure: a triangular connection
      f = 0.01;
      % 1. linear with mask
      sz1 = size(mask_);
      tfs{1}        = tf_conv_mask(mask_);
      tfs{1}.p(1).a = f*randn(sz1, 'single');
      tfs{1}.p(2).a = zeros(1,sz1(end), 'single');
      % 2. ReLu
      tfs{2}   = tf_relu();
      tfs{2}.i = tfs{1}.o;
      % 3. Dropout
      tfs{3}   = tf_dropout();
      tfs{3}.i = tfs{2}.o;
      % 4. linear
      tfs{4}        = tf_conv();
      tfs{4}.i      = tfs{3}.o;
      tfs{4}.p(1).a = f*randn(sz2, 'single');
      tfs{4}.p(2).a = zeros(1,sz2(end), 'single');
      % write back
      ob.tfs = tfs;
      
      %%% input&output data
      ob.i = n_data(); % X the feature
      ob.o = n_data(); % pout the prediction
      
      %%% collect the parameters
      ob.p = dag_util.collect_params( ob.tfs );
    end % tfw_rpd_reg
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a = ob.i.a; % [M, L, 1, N]
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs )
         ob.tfs{i} = fprop(ob.tfs{i});
         ob.ab.sync();
       end
       
       %%% Internal Output --> Outer Output
       % [1,1,2L, N] --> [2,L,N], from matconvnet format to internal format
       L = size(ob.i.a, 2);
       N = size(ob.i.a, 4);
       ob.o.a = reshape(ob.tfs{end}.o.a, [2, L, N] );
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      % [2,L,N] --> [1,1,2L,N]
      L = size(ob.o.d, 2);
      N = size(ob.o.d, 3);
      ob.tfs{end}.o.d = reshape(ob.o.d, [1,1,2*L,N] );
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop(ob.tfs{i});
        ob.ab.sync();
      end
      
      %%% Internal Input --> Outer Input: unnecessary here      
      ob.i.d = ob.tfs{1}.i.d;
    end % bprop
    
  end % methods
  
end

