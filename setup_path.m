%% config
dir_matconvnet = 'D:\CodeWork\git\psmatconvnet';
%% the matconvnet
% matconvnet/matlab
run( fullfile(dir_matconvnet, 'matlab\vl_setupnn') );
% matconvnet/matlab_dag
tmp = fileparts( mfilename('fullpath') );
cd( fullfile(dir_matconvnet, 'matlab_dag') );
eval( 'dag_path.setup' );
cd(tmp);
%% this
dir_this = fileparts( mfilename('fullpath') );
addpath( pwd );
