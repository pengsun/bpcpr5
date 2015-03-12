%%
dir_mo = 'D:\CodeWork\git\bpcpr5\script\T8\mo_zoo\T24';
it = 1 : 22;
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
  ii = 1 + mod(i-1,numel(spec));
  plot(s(i).ss(:), spec{ii});
  text(1, s(i).ss(1), num2str(i) );
%   plot( s(i).mag(:), spec{ii});
%   text(1, s(i).mag(1), num2str(i) );
  
end
hold off;