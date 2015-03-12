function save_and_leave(ob, evt)
%SAV_AND_LEAVE Summary of this function goes here
%   Detailed explanation goes here

t =  ob.cc.epoch_cnt;
tt = ob.cc.iter_cnt;
if (  ~(t==1 && tt==1) ), return; end

% convert to cpu data
ob.the_dag = to_cpu( ob.the_dag );

dir_mo = pwd;
% generate current model file name
fn_cur_mo = fullfile(...
  dir_mo, sprintf('ep%d_it%d.mat',t,tt) );
% save
fprintf('saving model %s...', fn_cur_mo);
save(fn_cur_mo, 'ob', '-v7.3');
fprintf('done\n');

error('done and terminate');

