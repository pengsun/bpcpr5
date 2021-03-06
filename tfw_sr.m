classdef tfw_sr < tfw_i
  %tfw_gpu_sr Stage Regressor: the feature extractor + a regressor
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = tfw_sr(hfet, hreg)
      %%% set the connection structure: a triangular connection
      % 1. multiplexer
      tfs{1} = tf_mtx(2);
      % 2. rpd feature
      tfs{2}      = hfet;
      tfs{2}.i(1) = tfs{1}.o(1);
      % 3. the regressor
      tfs{3}   = hreg;
      tfs{3}.i = tfs{2}.o;
      % 4. summer
      tfs{4}      = tf_add(2);
      tfs{4}.i(1) = tfs{3}.o;
      tfs{4}.i(2) = tfs{1}.o(2);
      % write back
      ob.tfs = tfs;
      
      %%% input&output data
      ob.i = [n_data(), n_data()]; % p the pose, I the image
      ob.o = n_data();             % pout the prediction
      
      %%% set the parameters
      ob.p = dag_util.collect_params( ob.tfs );
    end % tfw_rpd_reg
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{1}.i.a    = ob.i(1).a; % p [2,L,N]
       ob.tfs{2}.i(2).a = ob.i(2).a; % I [W,H,3,N]
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs ) 
         ob.tfs{i} = fprop(ob.tfs{i});
         ob.ab.sync();
       end
       
       %%% Internal Output --> Outer Output: 
       ob.o.a = ob.tfs{end}.o.a; % pPre [2,L,N]
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      ob.tfs{end}.o.d = ob.o.d; % dPre: [2,L,N]
         
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1 
        ob.tfs{i} = bprop(ob.tfs{i});
        ob.ab.sync();
      end
      
      %%% Internal Input --> Outer Input
      ob.i(1).d = ob.tfs{1}.i.d;    % [2,L,N]
      ob.i(2).d = ob.tfs{2}.i(2).d; % [W,H,3,N]
    end % bprop
         
  end % methods
  
end

