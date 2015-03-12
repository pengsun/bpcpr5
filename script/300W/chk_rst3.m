%%
dir_mo = pwd;
it = 1 : 32;
%%
for i = 1 : numel(it)

  ii = it(i);
  fn = sprintf('ep%d_it%d.mat', 1, ii);
  
  fprintf('processing %s...', fn);
  ffn = fullfile(dir_mo, fn);
  load(ffn);
  sa{i} = double( ap_std );
  sd{i} = double( dx_std );
  
  fprintf('done\n');
end
%%
figure;
spec = {...
  'ro-','bo-','go-','yo-','ko-','mo-',...
  'rx-','bx-','gx-','yx-','kx-','mx-'};
hold on;
for i = 1 : numel(sa)
  ii = 1 + mod(i-1,numel(spec));
  
%   plot(sa{i}, spec{ii});
%   text(1, sa{i}(1), num2str(i) );
  
  plot(sd{i}, spec{ii});
  text(1, sd{i}(1), num2str(i) );
  
end
hold off;