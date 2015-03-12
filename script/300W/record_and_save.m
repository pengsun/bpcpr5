function [ output_args ] = record_and_save(ob, evt)
%RECORD Summary of this function goes here
%   Detailed explanation goes here

ttt = tic;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t =  ob.cc.epoch_cnt;
tt = ob.cc.iter_cnt;
% if (  ~(t==1 && tt==1) ), return; end

[ap, ap_std] = chk.get_ap( ob.the_dag );
[dx, dx_std] = chk.get_dx( ob.the_dag );

dir_mo = pwd;
% generate current model file name
fn_cur_mo = fullfile(...
  dir_mo, sprintf('ep%d_it%d.mat',t,tt) );
% save
fprintf('saving statistics %s...', fn_cur_mo);
save(fn_cur_mo,...
  'ap','ap_std', 'dx','dx_std');
fprintf('done. ');
ttt = toc(ttt);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('time = %.3f\n', ttt);

end

