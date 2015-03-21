function hh = plot_face_lfpw(xx, yy, color, varargin)
%PLOT_FACE_LFPW Summary of this function goes here
%   hh = plot_face_lfpw(xx, yy, 'b', 'parent',hax);
%   hh = plot_face_lfpw(xx, yy, 'r', 'parent',hax);

%%% points
str = [color,'.'];
varargin = {str, varargin{:}};
% plot all points
hh = plot(xx,yy, varargin{:});


%%% segments
str = [color,'.-'];
varargin{1} = str;
% chin
ix = 1 : 17;
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% left eyebow
ix = 18 : 22;
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% right eyebow
ix = 23 : 27;
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% nose vertical
ix = 28 : 31;
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% nose horizontal
ix = 32 : 36;
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% left eye
ix = 37 : 42;
ix(end+1) = ix(1);
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% right eye
ix = 43 : 48;
ix(end+1) = ix(1);
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% mouth outer
ix = 49 : 60;
ix(end+1) = ix(1);
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});

% mouth inner
ix = 61 : 68;
ix(end+1) = ix(1);
hh(end+1) = plot(xx(ix), yy(ix), varargin{:});