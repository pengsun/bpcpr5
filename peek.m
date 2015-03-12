classdef peek < handle
  %peek Observer that do printing, plotting stuff
  %   Observer Design Pattern
  
  properties
    dir_mo;    % 
    is_cldata; % clear data?
    iter_mo;
  end
  
  methods
    function ob_this = peek()
      ob_this.dir_mo    = fileparts( mfilename );
      ob_this.is_cldata = true;
      ob_this.iter_mo   = 1;
    end
    
    function print_info(h, evt)
      %fprintf('epoch %d, batch %d of %d, ',...
      %  h.cc.epoch_cnt, h.cc.iter_cnt, hbat.num_bat);
      %fprintf('time = %.3fs, speed = %.0f images/s\n',...
      %  t_elapsed, ob.batch_sz/t_elapsed);
    end % print_bat
    
    function plot_loss(~, h, ~) % (ob_this, ob, evt)
      figure(42);
      yy = h.L_tr;
      xx = 1 : numel(yy);
      plot(xx,yy, 'bo-', 'linewidth',2);
      set(gca,'yscale','log');
      xlabel('epoch');
      ylabel('training loss');
      grid on;
      drawnow;
    end % print_bat
    
    function save_mo_ep_it(ob_this, ob, ~) % (ob_this, ob, evt)
      t =  ob.cc.epoch_cnt;     
      tt = ob.cc.iter_cnt;
      if ( mod(tt, ob_this.iter_mo) ~= 0  )
        return;
      end
        
      % prepare: clear the data
      if ( ob_this.is_cldata ), clear_im_data(ob); end
      
      % generate current model file name
      fn_cur_mo = fullfile(...
        ob_this.dir_mo, sprintf('ep%d_it%d.mat',t,tt) );
      % check if it exists
      if ( ~exist(ob_this.dir_mo, 'file') ), mkdir(ob_this.dir_mo); end
      % save
      fprintf('saving model %s...', fn_cur_mo);
      save(fn_cur_mo, 'ob', '-v7.3');
      fprintf('done\n');
    end % save_mo_ep_it
    
    function save_mo_ep(ob_this, ob, ~) % (ob_this, ob, evt)
      % prepare: clear the data
      if ( ob_this.is_cldata ), clear_im_data(ob); end
      
      % generate current model file name
      t =  ob.cc.epoch_cnt;  
      fn_cur_mo = fullfile(...
        ob_this.dir_mo, sprintf('ep%d.mat',t) );
      % check if it exists
      if ( ~exist(ob_this.dir_mo, 'file') ), mkdir(ob_this.dir_mo); end
      % save
      fprintf('saving model %s...', fn_cur_mo);
      save(fn_cur_mo, 'ob', '-v7.3');
      fprintf('done\n');
    end % save_mo_ep
  end % methods
  
end % classdef

