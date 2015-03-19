classdef convdag_bpcpr < handle
  %convdag CPR with Back Propagation
  %   A thin wrapper for the DAG, managing training and testing
  
  % options
  properties
    Nstar;     % #augmented size
    beg_epoch; % beggining epoch
    num_epoch; % number of epoches
    batch_sz;  % batch size
    is_tightMem; % if tight memory?
  end
  
  properties
    the_dag; % the whole DAG as a single transformer
    opt_arr; % numeric optimization array, one for each params(i)
    
    L_tr; % training loss
    cc; % calling context
  end
  
  events
    end_it; % end of one iteration (one batch)
    end_ep; % end of one epoch
  end
  
  methods
    function ob = convdag_bpcpr()
      ob.beg_epoch = 1; % begining epoch
      ob.num_epoch = 5; % number of epoches
      ob.batch_sz = 128; % batch size
      ob.is_tightMem = false;
      
      ob.cc = call_cntxt();
    end
    
    function ob = train (ob, X, Y)
    % train with instance X and lables Y
    % Input:
    %   X: [d1,d2,d3, N], where N = #instances, d1,...,dm are dims
    %   Y: [K, N], where K is #dims of the labels
    %
      
      %%% initialize the dag before calling train() 
      
      ob = prepare_train (ob);
      
      for t = ob.beg_epoch : ob.num_epoch
        % fire: train one epoch
        ob = prepare_train_one_epoch(ob, t);
        ob = train_one_epoch(ob, X,Y);
        ob = post_train_one_epoch(ob, t, ob.Nstar);
        
        notify(ob, 'end_ep');
      end % for t
      
    end % train
    
    function pPre = test (ob, pInit, I)
      
      % enforced single
      I = single(I);

      % prepare
      ob = prepare_test(ob);
      
      % initialize a batch generator
      hbat = bat_gentor();
      N = size(I, 4);
      hbat = reset(hbat, N, ob.batch_sz);
      
      % test every batch
      % What? Why dividing the testing set into batches? Becuuse this would
      % generate many print infos relieving you while you watch the screen
      for i_bat = 1 : hbat.num_bat
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
        % get batch 
        ind = get_idx_orig(hbat, i_bat);
        bat_pInit     = pInit(:,:,ind);
        bat_I         = I(:,:,:, ind);
        bat_trash_pGT = 0; % Just making the fprop() goes. Okay with a scalar.
        
        % set source nodes
        ob.the_dag.i(1).a = bat_pInit;
        ob.the_dag.i(2).a = bat_I;
        ob.the_dag.i(3).a = bat_trash_pGT;
        
        % fire: do the batch testing by calling fprop() on each transformer
        ob.the_dag = fprop( ob.the_dag );
        
        % fetch and concatenate the results
        bat_pPre = squeeze( ob.the_dag.get_pPre() );
        if (i_bat==1), pPre = bat_pPre;
        else           pPre = cat(3,pPre,bat_pPre); end
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        fprintf('testing: epoch %d, batch %d of %d, ',...
          ob.cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
      end % for ii
      
    end % test
  end % methods
    
  methods % auxiliary functions for train
    function ob = prepare_train (ob)

      % the parameters and the corresponding numeric optimizers
      num_params = numel(ob.the_dag.p);
      if ( numel(ob.opt_arr) ~= num_params )
        ob.opt_arr = opt_1storder();
        ob.opt_arr(num_params) = opt_1storder();
      end
      clear num_params;
      
      %%% set calling context
      % for the DAG
      ob.the_dag = set_cc(ob.the_dag, ob.cc);
      % for the optimizers
      for k = 1 : numel(ob.opt_arr)
        ob.opt_arr(k).cc = ob.cc;
      end
      % indicate it's the training stage
      ob.cc.is_tr = true;
      
    end % prepare_train
    
    %%% for training one epoch
    function ob = prepare_train_one_epoch (ob, i_epoch)
      % set calling context
      ob.cc.epoch_cnt = i_epoch;

      % update the loss
      ob.L_tr(i_epoch) = 0;
    end % prepare_train_one_epoch
    
    function ob = train_one_epoch (ob, X,Y)
    % train one epoch
      
      % enforced single
      X = single(X);
      Y = single(Y);
      
      % initialize a batch index generator
      hbat = batpose_gentor();
      N = size(X, 4);
      hbat = reset(hbat, N, ob.Nstar, ob.batch_sz);
      
      % train every batch
      for i_bat = 1 : hbat.num_bat
        
        t_bat = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % get batch 
        [bat_I,bat_pGT,bat_pInit] = get_data(hbat, X,Y, i_bat);
        t_bat = toc(t_bat); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        t_elapsed = tic; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % set source nodes
        ob.the_dag.i(1).a = bat_pInit;
        ob.the_dag.i(2).a = bat_I;
        ob.the_dag.i(3).a = bat_pGT;
        
        % fire: do the batch training
        ob = prepare_train_one_bat(ob, i_bat);
        ob = train_one_bat(ob);
        ob = post_train_one_bat(ob, i_bat);
        t_elapsed = toc(t_elapsed); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % print 
        % eopoch & iter
        fprintf('epoch %d, batch %d of %d, ',...
          ob.cc.epoch_cnt, i_bat, hbat.num_bat);
        fprintf('batch gen time = %.3fs ',...
          t_bat);
        fprintf('time = %.3fs, speed = %.0f images/s\n',...
          t_elapsed, ob.batch_sz/t_elapsed);
        
        %tmp = gpuDevice(1);
        %fprintf('gpu free memory = %d\n',...
        %  tmp.FreeMemory/1024/1024);
        
        notify(ob, 'end_it');

      end % for ii
    
    end % train_one_eporch
    
    function ob = post_train_one_epoch (ob, i_epoch, varargin)
      % normalize the loss
      N = varargin{1};
      ob.L_tr(end) = ob.L_tr(end) ./ N; 
    end % post_train_one_epoch
    
    %%% for traing one batch
    function ob = prepare_train_one_bat (ob, i_bat)
      % set calling context
      ob.cc.batch_sz = ob.batch_sz;
      ob.cc.iter_cnt = i_bat;
    end % prepare_train_one_bat
    
    function ob = train_one_bat (ob)
    % train one batch
    
      %%% fprop & bprop
      if (ob.is_tightMem)
        % TODO: modify it when vl_feat/matconvnet fully supports it
        warning('Tight memory has not been fully supported by original matconvnet!');
      end
      ob.the_dag = fprop( ob.the_dag );
      ob.the_dag = bprop( ob.the_dag );
      
      %%% update parameters
      for i = 1 : numel(ob.opt_arr)
        ob.opt_arr(i) = update(ob.opt_arr(i), ob.the_dag.p(i) );
      end
    end % train_one_bat
    
    function ob = post_train_one_bat (ob, i_bat)
      % update the loss
      LL = gather( ob.the_dag.o.a ); % cpu or gpu array
      ob.L_tr(end) = ob.L_tr(end) + sum(LL(:));
    end % post_train_one_bat
    
    %%% for internal data management
    function ob = clear_im_data (ob)
    % clear the intermediate (unnecessary) data: hidden variables .a, .d
    % parameters .d
      
      % clear the input for each transformer
      ob.the_dag = cl_io( ob.the_dag );
      
      % clear .d for all parameters
      % TODO: set a swith here, as sometimes we want save the gradients
      %ob.the_dag = cl_p_d( ob.the_dag );
      
    end % clear_im_data
     
  end % methods
     
  methods % auxiliary functions for test
    function ob = prepare_test(ob)
      ob.cc.is_tr = false;
    end % prepare_test
  end % methods    
  
end % convdag

