%% Initialization step       %%
%% Written by Ning Zhang     %%

% Add caffe matlab wrapper path.
% Change to your path
prcnn = config_prcnn;

% caffe
addpath(prcnn.dir.CAFFE_MATLAB_PATH);

% rcnn
addpath(prcnn.dir.RCNN_PATH);

% imdb
addpath('imdb_cub');

% Add liblinear package path
% Change to your path
addpath(genpath(prcnn.dir.LIBLINEAR_RCNN_PATH));

% eval(sprintf('run %s/startup', prcnn.dir.RCNN_PATH));

addpath('vis');
addpath('utils');
