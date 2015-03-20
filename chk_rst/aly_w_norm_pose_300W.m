function [yy, yy_ep] =  aly_w_norm_pose_300W(varargin)

% config 
t  = 3; % stage count
ep = 1 : 98;
dir_root = 'D:\CodeWork\git\bpcpr5';
dir_mo   = fullfile(dir_root,'\script\300W\mo\T24');
fn_mo_tmpl = 'ep%d.mat';

% plot
yy_ep = [];
yy    = [];
figure;
hax = axes;
title(dir_mo, 'Interpreter','none');

for i = 1 : numel(ep)
  iep = ep(i);
  if ( mod(iep-1,12)~=0 ), continue; end
  
  % init dag: from file 
  fn_mo = sprintf(fn_mo_tmpl, ep(i));
  ffn_mo = fullfile(dir_mo, fn_mo);
  if ( ~exist(ffn_mo,'file') )
    fprintf('%s not found, break and stop.\n', ffn_mo);
    break; 
  end
  load(ffn_mo, 'ob'); % get ob from here
  
 
  % get the norm
  w = ob.the_dag.tfs{1+t}.tfs{3}.tfs{4}.p(1).a;
  w = gather(w);
  yy{i} = get_w_norm_pose(w);

  % show norm
  plot_yy(hax, iep, yy{i})
  
end



function plot_yy(hax, iep, yyy)
str = {...
  'ro-','bo-','mo-','ko-','go-',...
  'rx-','bx-','mx-','kx-','gx-'...
};

ii = 1 + mod(iep-1, numel(str));

hold on;
plot(yyy(:), str{ii}, 'parent', hax);
text(1, yyy(1), num2str(iep),  'parent', hax);
hold off;
xlabel('point count');
ylabel('max norm for pose');
% set(hax, 'yscale','log');
grid on;
drawnow;