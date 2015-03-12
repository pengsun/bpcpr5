classdef chk
  %CHK Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Static)
    function [ap, ap_std] = get_ap( the_dag )
      h = the_dag;
      T = numel(h.tfs) - 2; % exclude mtx and loss
      for t = 1 : T
        tmp       = gather( h.tfs{1+t}.o.a );
        ap{t}     = tmp;
        ap_std(t) = std( tmp(:) ); 
      end % t
      
    end % get_ap
    
    function [dx, dx_std] = get_dx( the_dag )
      h = the_dag;
      T = numel(h.tfs) - 2; % exclude mtx and loss
      for t = 1 : T
        tmp       = gather( h.tfs{1+t}.tfs{2}.o.d );
        dx{t}     = tmp;
        dx_std(t) = std( tmp(:) ); 
      end % t
      
    end % get_dx
    
  end % methods(Static)
  
end

