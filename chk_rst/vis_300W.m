%% clear except
clearvars -except I p
%% config 
ep = 98;
dir_root   = 'D:\CodeWork\git\bpcpr5';
dir_data   = 'D:\data\facepose\300-Wnorm_matlab';
dir_mo     = fullfile(dir_root,'\script\300W\mo\T24');
fn_data    = fullfile(dir_data,'te_rescale_grad.mat');
fn_mo_tmpl = 'ep%d.mat';

%% load data
ind = 400;
if ( ~exist('I','var') || ~exist('p','var') )
  load(fn_data, 'I','p');
end
[bat_pInit,bat_I,bat_pGT] = get_te_data(I,p, ind);

%% init dag: from file
fn_mo = sprintf(fn_mo_tmpl, ep);
ffn_mo = fullfile(dir_mo, fn_mo);
if ( ~exist(ffn_mo,'file') )
  error('%s not found, break and stop.\n', ffn_mo);
end
load(ffn_mo, 'ob'); % get ob from here

%% test the instances with the model
ob.batch_sz = 2;
pPre = test(ob, bat_pInit, bat_I);
pPre = gather(pPre);

%% print the error
err = calc_pupil_dist(pPre, bat_pGT);
fprintf('data: %s\n', fn_data);
fprintf('model: %s\n', fn_mo);
fprintf('pupil distance = %d\n', err(end) );

%% visualize the results
% show the image
figure('WindowStyle','dock');
hax = axes;
II = bat_I(:,:,1,1);
imshow(II, []);
[H, W] = size(II); 

hold on;
% show the ground truth
plot(H*bat_pGT(1,:,1), W*bat_pGT(2,:,2),...
    'r.', 'markersize',12,...
    'parent',hax);
% show the stage 0
ps = gather( ob.the_dag.tfs{2}.i(1).a );
hp = plot(H*ps(1,:,1), W*ps(2,:,2),...
  'b.', 'markersize',12,...
  'parent',hax);
waitforbuttonpress;
% show the stages 1 to T
T = numel(ob.the_dag.tfs) - 2;
for i = 1 : T
  ps = gather( ob.the_dag.tfs{i+1}.o.a );
  delete(hp);
  hp = plot(H*ps(1,:,1), W*ps(2,:,2),...
    'b.', 'markersize',12,...
    'parent',hax);
  waitforbuttonpress;
end
hold off;
