classdef tfw_gpu_cpr < tfw_i
  %TFW_GPU_CPR Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    
    function ob = tfw_gpu_cpr()
      %#ok<*AGROW>
      %%% network connection
      T = 8;
      % The multiplexer for image I
      tfs{1} = tf_mtx(T);   
      % The T stage regressors
      for j = 1 : T        
        % the regressor along
        % TODO: the correct parameters
        the_mask = zeros(1,1,10,100);
        sz1 = [1,1,10,100];
        sz2 = [1,1,100,90];
        hreg = tfw_gpu_linloclin(the_mask, sz1, sz2);
        
        % the stage regressor (feature + regressor)
        tfs{1+j} = tfw_gpu_rpd_reg(hreg); 
        if (j>1)
          tfs{1+j}.i(1) = tfs{j}.o;  % p: connect to last stage
        end
        tfs{1+j}.i(2) = tfs{1}.o(j); % I: connect to the mtx 
      end
      % The loss
      tfs{2+T} = tf_loss_lse();
      tfs{2+T}.i(1) = tfs{1+T}.o; %
      % write back
      ob.tfs = tfs;
      
      
      %%% Input & Output data
      ob.i = [n_data(), n_data(), n_data()]; % pInit, I, pGT
      ob.o = n_data();                       % pPre the prediction
      
      
      %%% Collect the parameters
      ob.p = dag_util.collect_params( ob.tfs );
    end % tfw_gpu_cpr
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{2}.i(1).a   = ob.i(1).a; % pInit
       ob.tfs{1}.i.a      = ob.i(2).a; % I
       ob.tfs{end}.i(2).a = ob.i(3).a; % pGT
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs )
         ob.tfs{i} = fprop(ob.tfs{i});
         wait(gpuDevice);
       end
       
       %%% Internal Output --> Outer Output: set the loss
       %ob.o.a = ob.tfs{end}.o.a;      
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      %ob.tfs{end}.o.d = ob.o.d;
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1
        ob.tfs{i} = bprop(ob.tfs{i});
        wait(gpuDevice);
      end
      
      %%% Internal Input --> Outer Input: unnecessary here      
      %ob.i.d = ob.tfs{1}.i.d;
    end % bprop
    
  end % methods
  
end

