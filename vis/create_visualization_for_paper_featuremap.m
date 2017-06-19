%function create_visualization_for_paper
% visualization of feature map
%{=
prcnn = config_prcnn(1,10000);
prcnn_after = config_prcnn(2,20000);

% load imdb
load(prcnn.dir.IMDB_TEST_FILE);

% load model
load(prcnn.dir.TRAIN_MODEL_FILE);

% load predict results
load('predict_results.mat');
acnew = pd_new == ltest;
acold = pd_old == ltest;
inds = find(acnew==1 & acold==0);

Nsample = length(inds);
%}
%R=randperm(Nsample);

%% go to ./rcnn directory
% helper_create_visualization_featuremap;