function [err, err_ep] =  te_300W(varargin)

% config 
dir_root = 'D:\CodeWork\git\bpcpr5';
dir_data = 'D:\data\facepose\300-Wnorm_matlab';
if ( nargin==0 )
  ep = 1 : 14;
  batch_sz = 16;
  dir_mo = fullfile(dir_root,'\script\300W\mo\T24_aug200');
  fn_data = fullfile(dir_data,'te_rescale_grad.mat');
  fn_mo_tmpl = 'ep%d.mat';
elseif ( nargin==5 )
  ep = varargin{1};
  batch_sz = varargin{2};
  dir_mo = varargin{3};
  fn_data = varargin{4};
  fn_mo_tmpl = varargin{5};  
else
  error('Invalid arguments.');
end

% load data
[pInit,I,pGT] = load_te_data(fn_data);

% print
fprintf('data: %s\n', fn_data);

% plot
err_ep = 0;
err = 1;
figure;
hax = axes;
title(dir_mo, 'Interpreter','none');
plot_err(hax, err_ep, err);

for i = 1 : numel(ep)
  % init dag: from file 
  fn_mo = sprintf(fn_mo_tmpl, ep(i));
  ffn_mo = fullfile(dir_mo, fn_mo);
  if ( ~exist(ffn_mo,'file') )
    fprintf('%s not found, break and stop.\n', ffn_mo);
    break; 
  end
  load(ffn_mo, 'ob');
  % get ob from here
 
  ob.batch_sz = batch_sz;
  pPre = test(ob, pInit, I);
  pPre = gather(pPre);

  % show the error
  err(1+i) = calc_pupil_dist(pPre, pGT);
  err_ep = [err_ep, ep(i)];
  plot_err(hax, err_ep, err)
  
  % print the error
  fprintf('model: %s\n', fn_mo);
  fprintf('pupil distance = %d\n', err(end) );
end


function [pInit,I,pGT] = load_te_data(fn_data)
tmp = load(fn_data, 'I','p');
I   = tmp.I;
pGT = tmp.p;
clear tmp;

load('pMean_300W.mat', 'pMean');
N = size(pGT,3);
pInit = repmat(pMean, 1,1,N);


function plot_err(hax, err_ep, err)
plot(err_ep, err, 'ro-', 'linewidth', 2, 'parent', hax);
xlabel('epoches');
ylabel('testing pupil distance');
% set(hax, 'yscale','log');
grid on;
drawnow;