%%
clear;
fn = 'ep1_it6.mat';
ffn = fullfile(...
  'C:\Dev\code\bpcpr5\script\T8\mo_zoo\T24', fn);
s = get_iv(ffn);
%%
figure();
subplot(1,2,1);
plot(s.ss, 'ro-');
subplot(1,2,2);
plot(s.mag,'bo-');
title(fn, 'Interpreter','none');