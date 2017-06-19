%function create_visualization_for_paper
% visualization of ressultant classifiers
%{
prcnn = config_prcnn(1,10000);
prcnn_after = config_prcnn(2,20000);

% load imdb
% go to rcnn directory
% imdb_train = imdb_from_voc('train');
% imdb_test = imdb_from_voc('test');
% imdb_weak = imdb_from_voc('weak');
% roidb_train = roidb_from_voc(imdb_train);
rois_train = roidb_train.rois;

% load predict results
load('predict_results.mat');
acnew = pd_new == ltest;
acold = pd_old == ltest;
inds = find(acnew==1 & acold==0);


%% load pre-finetune results
% load model
load(prcnn.dir.TRAIN_MODEL_FILE);

% load train feature 
load(prcnn.dir.TRAIN_FEA_FILE);
TRN_fea = [];
for i = 1 : prcnn.config.N_parts
    TRN_fea = [TRN_fea train_fea{i}];
end
TRN_fea = scale_feature(TRN_fea);

% load test feature
load(prcnn.dir.TEST_FEA_FILE);
TST_fea = [];
for i = 1 : prcnn.config.N_parts
    TST_fea = [TST_fea test_fea{i, 2}];
end
TST_fea = scale_feature(TST_fea);

% load detect boxes
detect_boxes = get_rcnn_detections(prcnn);



%% load after-finetune results
% load model
model_after = load(prcnn_after.dir.TRAIN_MODEL_FILE);
model_after = model_after.model;

% load train feature 
load(prcnn_after.dir.TRAIN_FEA_FILE);
TRN_fea_after = [];
for i = 1 : prcnn.config.N_parts
    TRN_fea_after = [TRN_fea_after train_fea{i}];
end
TRN_fea_after = scale_feature(TRN_fea_after);

% load weak features
load(prcnn_after.dir.WEAK_FEA_FILE);
WEAK_fea_after = [];
for i = 1 : prcnn.config.N_parts
    WEAK_fea_after = [WEAK_fea_after weak_fea{i}];
end
WEAK_fea_after = scale_feature(WEAK_fea_after);

% load test feature
load(prcnn_after.dir.TEST_FEA_FILE);
TST_fea_after = [];
for i = 1 : prcnn.config.N_parts
    TST_fea_after = [TST_fea_after test_fea{i, 2}];
end
TST_fea_after = scale_feature(TST_fea_after);

% load detect boxes
detect_boxes_after = test_rcnn_parts_for_weak(prcnn);


Nsample = length(inds);

trainValid = cell(1,3);
for k=1:3
    trainValid{k} = find(sum(TRN_fea(:,(k-1)*4096+1:k*4096),2)>0);
end
%}


%R = [20 26 543 564 689 1914 2137 2243 5700 4416 4470 3896 2363 2367];
%Nsample = length(R);
%R=randperm(Nsample);
R = 1:Nsample;
for j=1:Nsample
    % before re-finetuning, only train
    try
        %i=inds(R(j));
        %i=R(j);
        i=2363;
        helper_for_visualization(i, prcnn, pd_old, imdb_test, imdb_train, model, ...
            TST_fea, TRN_fea, trainValid, detect_boxes, rois_train, 1);

        % after re-finetuning, only train
        helper_for_visualization(i, prcnn_after, pd_new, imdb_test, imdb_train, model_after, ...
            TST_fea_after, TRN_fea_after, trainValid, detect_boxes, rois_train, 2);

        % after re-finetuning, train and weak
        helper_for_visualization(i, prcnn_after, pd_new, imdb_test, imdb_train, model_after, ...
            TST_fea_after, TRN_fea_after, trainValid, detect_boxes, rois_train, 3,...
            imdb_weak, detect_boxes_after, WEAK_fea_after);
    catch
        fprintf('Something wrong for %d\n', i);
        continue;
    end
end