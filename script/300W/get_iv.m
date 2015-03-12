function s = get_iv( ffn )
%GET_IV Summary of this function goes here
%   Detailed explanation goes here
load(ffn);
p  = ob.the_dag.p;
oa = ob.opt_arr;
clear ob;
%
num = numel(p);
[rr,ss,mag] = deal([]);
for i = 1 : 2 : num
  a   = p(i).a;
  d   = p(i).d;
  del = oa(i).delta;
  
  % ratio
  rr(end+1) = norm(d(:)) / norm(a(:));
  ss(end+1) = norm(del(:)) / norm(a(:));
  
  % norm/magnitude
  mag(end+1) = norm(del(:));
end
s.rr = rr;
s.ss = ss;
s.mag = mag;
end

