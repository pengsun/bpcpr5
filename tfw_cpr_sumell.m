classdef tfw_cpr_sumell < tfw_i
  %TFW_CPR_SUMELL Cascade Pose Reg, sum of many intermediate losses
  %   The graph input:  pInit, I, pGT
  %   The graph output: ell
  
  properties
  end
  
  methods
    
    function ob = tfw_cpr_sumell(tfs_cprpell)
    % Input: array of tfw_cpr_pell (two outputs)  
      %#ok<*AGROW>
      
      %%% network connection
      % number of stage regressors
      T = numel(tfs_cprpell);
      % The multiplexer for ground truth pGT
      tfs{1} = tf_mtx(T);
      % The multiplexer for image I
      tfs{2} = tf_mtx(T);   
      % The T cpr (each outputting p, ell)
      for j = 1 : T
        % the CPR
        tfs{2+j} = tfs_cprpell{j}; 
        if (j>1)
          tfs{2+j}.i(1) = tfs{1+j}.o(1); % p: connect to previous stage
        end
        tfs{2+j}.i(2) = tfs{2}.o(j);     % I: connect to the mtx
        tfs{2+j}.i(3) = tfs{1}.o(j);     % pGT: connect to the mtx
      end
      % The sum (of loss)
      tfs{2+T+1}   = tf_add(T);
      for j = 1 : T
        tfs{2+T+1}.i(j) = tfs{2+j}.o(2); 
      end
      % write back
      ob.tfs = tfs;
      
      %%% Input & Output data
      ob.i = [n_data(), n_data(), n_data()]; % pInit, I, pGT
      ob.o = n_data();                       % ell
      
      %%% Collect the parameters
      ob.p = dag_util.collect_params( ob.tfs );
    end % tfw_cpr_pell
    
    function ob = fprop(ob)
       %%% Outer Input --> Internal Input
       ob.tfs{3}.i(1).a = ob.ab.cvt_data( ob.i(1).a ); % pInit
       ob.tfs{2}.i.a    = ob.ab.cvt_data( ob.i(2).a ); % I
       ob.tfs{1}.i.a    = ob.ab.cvt_data( ob.i(3).a ); % pGT
       
       %%% fprop for all
       for i = 1 : numel( ob.tfs ) % sync() is called in sub tf
         ob.tfs{i} = fprop(ob.tfs{i});
       end
       
       %%% Internal Output --> Outer Output: set the loss
       ob.o.a = ob.tfs{end}.o.a; % ell
    end % fprop
       
    function ob = bprop(ob)
      %%% Outer output --> Internal output
      %ob.tfs{end}.o.d = ob.o.d; % ell
      %%% A trick for the sink node; Fine with the scalar
      ob.tfs{end}.o.d      = ob.ab.cvt_data( 1.0 ); % ell
      ob.tfs{end-1}.o(1).d = ob.ab.cvt_data( 0.0 ); % pPre
      
      %%% bprop for all
      for i = numel(ob.tfs) : -1 : 1 % sync() is called in sub tf
        ob.tfs{i} = bprop(ob.tfs{i});
      end
      
      %%% Internal Input --> Outer Input  
      ob.i(1).d = ob.tfs{3}.i(1).d; % pInit
      ob.i(2).d = ob.tfs{2}.i.d ;   % I
      ob.i(3).d = ob.tfs{1}.i.d;    % pGT (unnecessary?)
    end % bprop
    
    function pPre = get_pPre(ob)
    % a shorthand to get the prediction
      pPre = ob.tfs{end-1}.o(1).a;
    end % get_pPre
  end % methods
  
end

