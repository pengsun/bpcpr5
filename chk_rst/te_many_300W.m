function [err, err_ep] =  te_many_300W(varargin)

%%% config 
dir_root = 'D:\CodeWork\git\bpcpr5\script\300W\mo\';
dir_data = 'D:\data\facepose\300-Wnorm_matlab';

ep = 1 : 50;
batch_sz = 17;
moname = {...
  'T24_sumell_stg6_aug200',...
  'T24_sumell_stg6_aug200_eta0.001',...
  'T24_sumell_stg12_aug100'
};
dir_mo = cellfun(@(z) (fullfile(dir_root,z)), moname,...
  'uniformoutput',false);
fn_data = fullfile(dir_data,'te_cha_rescale_grad.mat');
fn_mo_tmpl = 'ep%d.mat';

%%% load data
fprintf('loading data %s...', fn_data);
[pInit,I,pGT] = load_te_data(fn_data);
fprintf('done\n');

%%% the vars
[err, err_ep, hp] = deal( cell(numel(moname),1) );

%%% plot
figure;
hax = axes;
for j = 1 : numel(moname)
  plot_mo_dir(j);
end
hhh = legend( moname );
set(hhh, 'Interpreter','none');

  function plot_mo_dir(ind_dirmo)
    fprintf('testing for model %s...\n', moname{ind_dirmo} );
    
    for i = 1 : numel(ep)
      % init dag: from file 
      ffn_mo = fullfile(dir_mo{ind_dirmo},...
                        sprintf(fn_mo_tmpl, ep(i)) );
      if ( ~exist(ffn_mo,'file') )
%         fprintf('%s not found, break and stop.\n', ffn_mo);
%         break; 
        fprintf('%s not found, continue.\n', ffn_mo);
        continue;
      end
      
      % calculate the error
      err{ind_dirmo}    = [err{ind_dirmo}, calc_err(ffn_mo)];
      err_ep{ind_dirmo} = [err_ep{ind_dirmo}, ep(i)];
      plot_err(ind_dirmo);

      % print the error
      fprintf('pupil distance = %d\n', err{ind_dirmo}(end) );
    end
  end % plot_mo_dir

  function er = calc_err(ffn_mo)
    % get ob from here
    load(ffn_mo, 'ob');
    
    ob.batch_sz = batch_sz;
    pPre = test(ob, pInit, I);
    pPre = gather(pPre);
    
    % show the error
    er = calc_pupil_dist(pPre, pGT);
  end

  function plot_err(ind_dirmo)
    % the line style
    sty = {...
      'ro-','bo-','go-','mo-','ko-',...
      'bp-','gp-','mp-','kp-','rp-',};
    ii = 1 + mod(ind_dirmo-1, numel(sty));
    
    % plot the line
    delete( hp{ind_dirmo} );
    hold on;
    hp{ind_dirmo} = plot(...
      err_ep{ind_dirmo}, err{ind_dirmo},...
      sty{ii}, 'linewidth', 2, 'parent', hax);
    hold off;
    
    % trim
    xlabel('epoches');
    ylabel('testing pupil distance');
    % set(hax, 'yscale','log');
    grid on;
    drawnow;
  end % plot_err

  function [pInit,I,pGT] = load_te_data(fn_data)
    tmp = load(fn_data, 'I','p');
    I   = tmp.I;
    pGT = tmp.p;
    clear tmp;

    load('pMean_300W.mat', 'pMean');
    N = size(pGT,3);
    pInit = repmat(pMean, 1,1,N);
  end % load_te_data

end