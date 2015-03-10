function tr()
%% init dag: from file or from scratch
beg_epoch = 4;

dir_root = fileparts( fileparts(mfilename) );
dir_mo   = fullfile(dir_root, 'mo_zoo', 'T8');
fn_mo = fullfile(dir_mo, sprintf('ep%d_it%d.mat', beg_epoch-1, 30) );
if ( exist(fn_mo, 'file') )
  h = create_dag_from_file (fn_mo);
else
  beg_epoch = 1; 
  h = create_dag_from_scratch ();
end
%% config for training 
h.beg_epoch = beg_epoch;
h.Nstar = 3148*20;
h.num_epoch = 200;
h.batch_sz = 128;
h.dir_mo = dir_mo;
fn_data  = fullfile(...
  '/home/ubuntu/A/data/facepose/300-Wnorm_matlab',... % directory 
  'tr_rescale_grad.mat');                             % file name
%% cpu or gpu
h = to_cpu(h);
% h = to_gpu(h);
%% do the training
[X, Y] = load_tr_data(fn_data);
train(h, X,Y);

function h = create_dag_from_scratch ()
tfs_sr = create_tfs_sr();
hdag = tfw_cpr(tfs_sr);
h = convdag();
h.the_dag = hdag;

function tfs_sr = create_tfs_sr()
% stage regressor array
T = 8;
for j = 1 : T
  % the feature extractor
  hfet = tf_fet_rpd();
  
  % the regressor
  % TODO: the correct params
  the_mask = zeors(1,1,3,7, 'single');
  sz1      = [1,1,1,1];
  sz2      = [1,1,1,1];
  hreg = tfw_linloclin(the_mask, sz1, sz2);
  
  % the stage regressor (feature + regressor)
  tfs_sr{j} = tfw_sr(hfet, hreg); %#ok<AGROW>
end

function ob = create_dag_from_file (fn_mo)
load(fn_mo, 'ob');
% ob loaded and returned

function [X,Y] = load_tr_data(fn_data)
tmp = load(fn_data);
X = tmp.I;
Y = tmp.p;
clear tmp;