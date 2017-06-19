function prcnn = config_prcnn(MODE_ID, ITER_NUM)
% configuration file
% MODE_ID: 1: training, 2: weak
% ITER_NUM: # of iterations used in finetuning caffe models
if nargin<1
    MODE_ID = 1;        % default: train mode
end

if nargin<2
    ITER_NUM = 10000;   % default number of iterations for fine-tuning cnn
end

% modes
prcnn.modes = {'train', 'weak'};
prcnn.mode = MODE_ID;
prcnn.iter = ITER_NUM;

%% global paths
prcnn.dir.PRCNN_PATH = '/home/zxu/workspace/part-based-RCNN-master';  % prcnn path
prcnn.dir.CAFFE_PATH = '/home/zxu/workspace/caffe-0.999';                   % caffe path
prcnn.dir.RCNN_PATH = '/home/zxu/workspace/part-based-RCNN-master/rcnn';   % rcnn path
prcnn.dir.BIRD_DIR = '/home/zxu/Dataset/CUB_200_2011/';               % cub dataset dir
prcnn.dir.WEAR_DIR = '/home/zxu/workspace/Flickr/data/images/';       % web data dir


%% other paths
% figure results and finetune model path
if MODE_ID == 1
    mode_type = '';
else
    mode_type = '-weak';
end
prcnn.dir.figure_dir = sprintf('results/figure%s', mode_type);
prcnn.dir.LIBLINEAR_RCNN_PATH = fullfile(prcnn.dir.PRCNN_PATH, 'liblinear-1.96'); % liblinear path
prcnn.dir.CACHE_PATH = fullfile(prcnn.dir.PRCNN_PATH, 'caches');      % cache path
%prcnn.dir.CACHE_PATH = fullfile(prcnn.dir.PRCNN_PATH, 'caches_weak_all_images');

%% caffe paths
prcnn.dir.CAFFE_FINETUNE_PATH = fullfile(prcnn.dir.CAFFE_PATH, ...                % caffe finetuning path
     sprintf('examples/cub-finetuning%s',mode_type));
%    'examples/cub-finetuning-weak(allweakimages)');


prcnn.dir.CAFFE_NET_FILE = fullfile(prcnn.dir.CAFFE_FINETUNE_PATH, ...
    ['cub_finetune_%s_iter_' sprintf('%d.caffemodel', ITER_NUM)]);
prcnn.dir.CAFFE_MATLAB_PATH = fullfile(prcnn.dir.CAFFE_PATH, 'matlab/caffe/');    % caffe matlab path

% mean file
prcnn.dir.CAFFE_MEAN_FILE = fullfile(prcnn.dir.RCNN_PATH, 'ilsvrc_2012_mean.mat');


%% rcnn paths
prcnn.dir.RCNN_CACHE_DIR = fullfile(prcnn.dir.RCNN_PATH, 'cachedir');
% imdb file paths
prcnn.dir.IMDB_TRAIN_FILE = fullfile(prcnn.dir.RCNN_PATH, 'imdb/cache/cub_parts_train.mat');
prcnn.dir.IMDB_TEST_FILE = fullfile(prcnn.dir.RCNN_PATH, 'imdb/cache/cub_parts_test.mat');
prcnn.dir.IMDB_WEAK_FILE = fullfile(prcnn.dir.RCNN_PATH, 'imdb/cache/cub_parts_weak.mat');
%prcnn.dir.IMDB_WEAK_FILE = fullfile(prcnn.dir.PRCNN_PATH, 'imdb_weak.mat');

% roidb file paths
prcnn.dir.ROIDB_TRAIN_FILE = fullfile(prcnn.dir.RCNN_PATH, 'imdb/cache/roidb_parts_train.mat');
prcnn.dir.ROIDB_TEST_FILE = fullfile(prcnn.dir.RCNN_PATH, 'imdb/cache/roidb_parts_test.mat');

% path for rcnn features
prcnn.dir.RCNN_FEATURE_PATH = fullfile(prcnn.dir.RCNN_PATH, ...
    ['feat_cache/v1_finetune_cub_train_%s_iter_' sprintf('%dk',ITER_NUM/1000)]);
% detect score file
prcnn.dir.RCNN_DETECT_SCORE_FILE = fullfile(prcnn.dir.RCNN_FEATURE_PATH, ...
     '%s/%s_score.mat');
 
% pool5 feature file
prcnn.dir.RCNN_POOL5_FILE = fullfile(prcnn.dir.RCNN_FEATURE_PATH, ...
     '%s/%s.mat');

% RCNN detection result file for each image in the weak dataset
prcnn.dir.WEAK_DETECT_BOX_IMAGE_FILE = fullfile(prcnn.dir.RCNN_FEATURE_PATH, ...
    '%s/%s_pdbox.mat');
prcnn.dir.WEAK_DETECT_BOX_IMAGE_MIL_FILE = fullfile(prcnn.dir.RCNN_FEATURE_PATH, ...
    '%s/%s_pdbox_mil.mat');

% selective search file
prcnn.dir.SELECTIVE_SEARCH_FILE = fullfile(prcnn.dir.RCNN_PATH, ...
    'data/selective_search_data/%s.mat');

% cached pool5 features for ground truth
prcnn.dir.GT_POS_FEATURE_FILE = fullfile(prcnn.dir.RCNN_PATH, ...
    'feat_cache/%s/%s/gt_pos_layer_5_cache.mat');

% pool5 feature stats file
prcnn.dir.RCNN_FEATURE_STAT_FILE = fullfile(prcnn.dir.RCNN_CACHE_DIR, ...
    '%s/feature_stats_%s_layer_%d_%s.mat');

% RCNN result of detecting model path
prcnn.dir.RCNN_DETECT_MODEL_FILE = fullfile(prcnn.dir.RCNN_CACHE_DIR, ...
    '%s/rcnn_model_%s.mat');



%% prcnn paths
% config
try
    load(fullfile(prcnn.dir.CACHE_PATH, 'cub2011_config.mat'));
catch
    eval(sprintf('run %s/get_bird_data', prcnn.dir.PRCNN_PATH));
    config = ans;
    save(fullfile(prcnn.dir.CACHE_PATH, 'cub2011_config.mat'), 'config');
end
prcnn.config = config;

% model definition for extracting features
prcnn.dir.MODEL_DEF_FILE = fullfile(prcnn.dir.CACHE_PATH, 'cub_finetune_deploy_fc7.prototxt');

% cnn model file
% for k=1:prcnn.config.N_parts
%     prcnn.dir.CNN_MODELS{k} = ...
%         sprintf('%s/cub_finetune_%s_iter_%d.caffemodel', ...
%         prcnn.dir.CAFFE_FINETUNE_PATH, prcnn.config.parts{k}, ITER_NUM);
% end
parts = {'bbox','head','body'};
for k=1:3
    % if you use cached models, uncomment this line
    %prcnn.dir.CNN_MODELS{k} = sprintf('./caches/CUB_%s_finetune_mod%d.caffe_model', parts{k}, MODE_ID);
    
    % **************************************** %
    % if you finetune caffe by your own using the method we suggest, use
    % uncomment the following tow sline
    prcnn.dir.CNN_MODELS{k} = sprintf('%s/cub_finetune_%s_iter_%d.caffemodel', ...
       prcnn.dir.CAFFE_FINETUNE_PATH, parts{k}, ITER_NUM);
end

% training model file
prcnn.dir.TRAIN_MODEL_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
        sprintf('finetune_model_fc7_%s_%d.mat', prcnn.modes{MODE_ID}, ITER_NUM));
    
prcnn.dir.TRAIN_MODEL_MIL_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
        sprintf('finetune_model_fc7_%s_%d_mil.mat', prcnn.modes{MODE_ID}, ITER_NUM));    
    
% training feature file
prcnn.dir.TRAIN_FEA_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
        sprintf('finetune_train_fea_fc7_%s_%d.mat', prcnn.modes{MODE_ID}, ITER_NUM));
% testing feature file    
prcnn.dir.TEST_FEA_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
        sprintf('finetune_test_fea_fc7_%s_%d.mat', prcnn.modes{MODE_ID}, ITER_NUM));    
% RCNN detection result file
prcnn.dir.DETECT_BOX_FILE = fullfile(prcnn.dir.CACHE_PATH, 'rcnn_detect_boxes.mat');

% weak feature file
prcnn.dir.WEAK_FEA_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
    sprintf('finetune_weak_fea_fc7_%s_%d.mat', prcnn.modes{MODE_ID}, ITER_NUM));

prcnn.dir.WEAK_FEA_MIL_FILE = fullfile(prcnn.dir.CACHE_PATH, ...
    sprintf('finetune_weak_fea_fc7_%s_%d_mil.mat', prcnn.modes{MODE_ID}, ITER_NUM));

% RCNN detection result file for the whole weak dataset
prcnn.dir.WEAK_DETECT_BOX_FILE = fullfile(prcnn.dir.CACHE_PATH, 'rcnn_detect_boxes_weak.mat');
prcnn.dir.WEAK_DETECT_BOX_MIL_FILE = fullfile(prcnn.dir.CACHE_PATH, 'rcnn_detect_boxes_weak_mil.mat');
% selected patches in weak set to be inserted to training set
prcnn.dir.SELECTED_WEAK_PATCH_FILE = fullfile(prcnn.dir.CACHE_PATH, 'selected_weak_samples.mat');

% geometric constraint file
prcnn.dir.GEO_PRIOR_FILE = fullfile(prcnn.dir.CACHE_PATH, 'geo_prior.mat');

% mil model file
prcnn.dir.MIL_MODEL_FILE = fullfile(prcnn.dir.CACHE_PATH, 'mil_model.mat');



%% parameters
prcnn.para.svm_option = '-s 1 -c 1 -q';  % svm option
prcnn.para.geo_box_gap = 10;            % for box geometric constraint, gap outside box
prcnn.para.geo_mg_alpha = .05;          % alpha for mixture gaussian geometric constraint
prcnn.para.isvisualize = 0;             % if needs to visualize results for detecting
prcnn.para.canSkip = 1;

prcnn.para.numSelfPacedLearning = 1;    % number of iterations of self-paced learning
prcnn.para.K0SelfPacedLearning = 1e5;   % initial K of self-paced learning, a very big number indicate no self-paced learning
prcnn.para.etaSelfPacedLearning = 10;   % eta of self-paced learning

prcnn.para.selected_geo_method = 2;     % selected method of geometric constraint for detecting on weak dataset

prcnn.para.isuse_mil = 1;               % if use multi-instance learning methods
end
