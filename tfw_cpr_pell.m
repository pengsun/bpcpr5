classdef tfw_cpr_pell < tfw_i
  %TFW_GPU_CPR_PELL Cascade Pose Reg, outputting pose (p) and loss (ell)
  %   The graph input:  pInit, I, pGT
  %   The graph output: pPre, ell
  
  properties
  end
  
  methods
    
    function ob = tfw_cpr_pell(tfs_sr)
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
      % The multiplexer for pose p
      tfs{2+T}   = tf_mtx(2);
      tfs{2+T}.i = tfs{1+T}.o; 
      % The loss
      tfs{3+T}      = tf_loss_lse();
      tfs{3+T}.i(1) = tfs{2+T}.o(2); % p
      % write back
      ob.tfs = tfs;
      
      %%% Input & Output data
      ob.i = [n_data(), n_data(), n_data()]; % pInit, I, pGT
      ob.o = [n_data(), n_data()];           % pPre, ell
      
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
       ob.o(1).a = ob.tfs{end-1}.o(1).a; % pPre
       ob.o(2).a = ob.tfs{end}.o.a;      % ell
    end % fprop
       
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      ob.tfs{end-1}.o(1).d = ob.o(1).d; % pPre
      ob.tfs{end}.o.d      = ob.o(2).d; % ell
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1 % sync() is called in sub tf
        ob.tfs{i} = bprop(ob.tfs{i});
      end
      
      %%% Internal Input --> Outer Input  
      ob.i(1).d = ob.tfs{2}.i(1).d;   % pInit
      ob.i(2).d = ob.tfs{1}.i.d ;     % I
      ob.i(3).d = ob.tfs{end}.i(2).d; % pGT (unnecessary?)
    end % bprop
    
    function pPre = get_pPre(ob)
    % a shorthand to get the prediction
      pPre = ob.o(1).a;
    end % get_pPre
  end % methods
  
end

