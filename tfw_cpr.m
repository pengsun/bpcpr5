classdef tfw_cpr < tfw_i
  %TFW_GPU_CPR Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
    
    function ob = tfw_cpr(tfs_sr)
      %#ok<*AGROW>
      
      %%% network connection
      % number of stage regressors
      T = numel(tfs_sr);
      % The multiplexer for image I
      tfs{1} = tf_mtx(T);   
      % The T stage regressors
      for j = 1 : T
        % the stage regressor (feature + regressor)
        tfs{1+j} = tfs_sr{j}; 
        if (j>1)
          tfs{1+j}.i(1) = tfs{j}.o;  % p: connect to previous stage
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
       ob.tfs{2}.i(1).a   = ob.ab.cvt_data( ob.i(1).a ); % pInit
       ob.tfs{1}.i.a      = ob.ab.cvt_data( ob.i(2).a ); % I
       ob.tfs{end}.i(2).a = ob.ab.cvt_data( ob.i(3).a ); % pGT
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs ) % sync() is called in sub tf
         ob.tfs{i} = fprop(ob.tfs{i});
       end
       
       %%% Internal Output --> Outer Output: set the loss
       ob.o.a = ob.tfs{end}.o.a;      
    end % fprop
    
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      %ob.tfs{end}.o.d = ob.o.d;
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1 % sync() is called in sub tf
        ob.tfs{i} = bprop(ob.tfs{i});
      end
      
      %%% Internal Input --> Outer Input: unnecessary here      
      %ob.i.d = ob.tfs{1}.i.d;
    end % bprop
    
  end % methods
  
end

