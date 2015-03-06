classdef tfw_gpu_rpd_reg < tfw_i
  %TFW_RPD_REG RPD feature + a regressor
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    function ob = tfw_gpu_rpd_reg(hreg)
      %%% set the connection structure: a triangular connection
      % 1. multiplexer
      tfs{1} = tf_mtx(2);
      % 2. rpd feature
      tfs{2}      = tf_fet_rpd();
      tfs{2}.i(1) = tfs{1}.o(1);
      % 3. the regressor
      tfs{3} = hreg;
      tfs{3}.i = tfs{2}.o;
      % 4. summer
      tfs{4} = tf_add(2);
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
       ob.tfs{1}.i.a    = ob.i(1).a; % p
       ob.tfs{2}.i(2).a = ob.i(2).a; % I
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs )
         ob.tfs{i} = fprop(ob.tfs{i});
         wait(gpuDevice);
       end
       
       %%% Internal Output --> Outer Output: set the loss
       ob.o.a = ob.tfs{end}.o.a;      
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      ob.tfs{end}.o.d = ob.o.d;
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop(ob.tfs{i});
        wait(gpuDevice);
      end
      
      %%% Internal Input --> Outer Input: unnecessary here      
      ob.i(1).d = ob.tfs{1}.i.d;
      ob.i(2).d = ob.tfs{2}.i(2).d;
    end % bprop
    
  end % methods
  
end

