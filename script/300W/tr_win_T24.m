function tr_win_T24()
%% config: data dir
fn_data  = fullfile(...
  'D:\data\facepose\300-Wnorm_matlab',... % directory
  'tr_rescale_grad.mat');                 % file name
%% config: model dir
dir_root = pwd;
dir_mo   = fullfile(dir_root, 'mo', 'T24_eta0.1');
%% init dag: from saved model 
% beg_epoch = 40;
% % fn_mo = fullfile(dir_mo, sprintf('ep%d_it%d.mat', beg_epoch-1, 30) );
% fn_mo = fullfile(dir_mo, sprintf('ep%d.mat', beg_epoch-1) );
% h = create_dag_from_file (fn_mo);
%% init dag: from scratch
beg_epoch = 1; 
h = create_dag_from_scratch ();
%% config: for training algorithm
h.beg_epoch = beg_epoch;
h.Nstar = 3148*20;
h.num_epoch = 200;
h.batch_sz = 32;
%% config: the optimization algorithms
eta = 1e-1;
h.opt_arr = opt_1storder();
for i = 1 : numel( h.the_dag.p )
  h.opt_arr(i)     = opt_1storder();
  h.opt_arr(i).eta = eta;
end

%% config: cpu or gpu
% h.the_dag = to_cpu(h.the_dag);
h.the_dag = to_gpu(h.the_dag);
%% add observers
hpeek = peek();
%%% plot
% addlistener(h, 'end_ep', @hpeek.plot_loss);

%%% save model epoch and itearation
hpeek.dir_mo = dir_mo;
% hpeek.iter_mo = 1500;
% addlistener(h, 'end_it', @hpeek.save_mo_ep_it);

%%% save model epoch
addlistener(h, 'end_ep', @hpeek.save_mo_ep);
%% do the training
[X, Y] = load_tr_data(fn_data);
train(h, X,Y);

function h = create_dag_from_scratch ()

fprintf('create_dag_from_scratch...');
h = convdag_bpcpr();
tfs_sr    = create_tfs_sr();
hdag      = tfw_cpr(tfs_sr);
h.the_dag = hdag;
fprintf('done\n');


function tfs_sr = create_tfs_sr()
%%% params for stage regressor array
% #stages
T = 24;
% for feature extractor 
MM = 15 * ones(1, T);  % #RPD features per point
rr = [ 0.09*ones(1,6), 0.04*ones(1,18)]; % radius
% for regressor
m = 6; % for hidden layers
K = 68;
knn = [5*ones(1,6), 3*ones(1,12), 1*ones(1,6)]; % for connection mask
% knn = 3*ones(1,T);

for j = 1 : T
  % the feature extractor
  Z = get_Z();
  hfet   = tf_fet_rpdni_mex(Z);
  hfet.r = rr(j);
  hfet.M = MM(j);
  
  % the regressor
  % local connection mask
  if     (K==34), A = get_mask_lfpw_K34( MM(j), m, knn(j) ); 
  elseif (K==68), A = get_mask_lfpw_K68( MM(j), m, knn(j) ); 
  end
  % [ML, mK] --> [M,L,1,mK]
  L = size(Z,1);
  A = reshape(A, [MM(j),L, 1, m*K]); % for matconvnet format
  % the output pose: 
  sz2 = [1, 1, m*K, 2*L]; % for matconvnet format
  hreg = tfw_linloclin(A, sz2);
  
  % the stage regressor (feature + regressor)
  tfs_sr{j} = tfw_sr(hfet, hreg); %#ok<AGROW>
end

function pMean = get_pMean ()
tmp = load('pMean_300W.mat');
pMean = tmp.pMean;

function Z = get_Z ()
tmp = load('lfpw_randpair_r0.20.mat');
Z = tmp.Z;

function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function [X,Y] = load_tr_data(fn_data)
fprintf('loading data %s...', fn_data);
tmp = load(fn_data);
X = tmp.I;
Y = tmp.p;
clear tmp;
fprintf('done\n');