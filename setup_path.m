%% config
dir_matconvnet = 'D:\CodeWork\git\matconvnet';
dir_matconvdag = 'D:\CodeWork\git\MatConvDAG';
%% matconvnet
run( fullfile(dir_matconvnet, 'matlab\vl_setupnn') );
%% matconvDAG
tmp = fileparts( mfilename('fullpath') );
cd( fullfile(dir_matconvdag, 'core') );
eval( 'dag_path.setup' );
cd(tmp);
%% this
% root
dir_this = fileparts( mfilename('fullpath') );
addpath( pwd );
% util
addpath( fullfile(pwd, 'util') );
% cache
addpath( fullfile(pwd, 'cache') );
