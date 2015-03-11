function tr_win()
%% config: data dir
fn_data  = fullfile(...
  'D:\data\facepose\300-Wnorm_matlab',... % directory
  'tr_rescale_grad.mat');                 % file name
%% config: model dir
dir_root = fileparts( fileparts(mfilename) );
dir_mo   = fullfile(dir_root, 'mo_zoo', 'T6');
%% init dag: from saved model 
% beg_epoch = 4;
% fn_mo = fullfile(dir_mo, sprintf('ep%d_it%d.mat', beg_epoch-1, 30) );
% h = create_dag_from_file (fn_mo);
%% init dag: from scratch
beg_epoch = 1; 
h = create_dag_from_scratch ();
%% config: for training algorithm
h.beg_epoch = beg_epoch;
h.Nstar = 3148*20;
h.num_epoch = 200;
h.batch_sz = 128;
h.dir_mo = dir_mo;
h.iter_mo = 1;
%% config: cpu or gpu
h.the_dag = to_cpu(h.the_dag);
% h.the_dag = to_gpu(h.the_dag);
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
T = 6;
% for feature extractor 
MM = 15 * ones(1, T);  % #RPD features per point
rr = [ 0.2*ones(1,12), 0.1*ones(1,12)]; % radius
% for regressor
m = 6; % for hidden layers
K = 34;
knn = [5*ones(1,8), 3*ones(1,8), 1*ones(1,8)]; % for connection mask

for j = 1 : T
  % the feature extractor
  pMean = get_pMean();
  hfet   = tf_fet_rpd(pMean);
  hfet.r = rr(j);
  hfet.M = MM(j);
  
  % the regressor
  % local connection mask
  if     (K==34), A = get_mask_lfpw_K34( MM(j), m, knn(j) ); 
  elseif (K==68), A = get_mask_lfpw_K34( MM(j), m, knn(j) ); 
  end
  % [ML, mK] --> [M,L,1,mK]
  L = size(pMean,2);
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