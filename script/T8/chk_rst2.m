%%
dir_mo = 'C:\Dev\code\bpcpr5\script\T8\mo_zoo\T6';
it = 1 : 10;
%%
for i = 1 : numel(it)
  ii = it(i);
  fn = sprintf('ep%d_it%d.mat', 1, ii);
  ffn = fullfile(dir_mo, fn);
  
  fprintf('processing %s...', fn);
  s(i) = get_iv( ffn );
  fprintf('done\n');
end
%%
figure;
spec = {...
  'ro-','bo-','go-','yo-','ko-','mo-',...
  'rx-','bx-','gx-','yx-','kx-','mx-'};
hold on;
for i = 1 : numel(s)
  plot( s(i).mag(:), spec{i});
  %plot( s(i).ss(:), spec{i});
end
hold off;