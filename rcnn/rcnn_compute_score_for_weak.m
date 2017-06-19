function rcnn_compute_score_for_weak(imdb, detector_model, varargin)
%  rcnn_compute_score(imdb, varargin)
%   Computes score using fc7 features and rcnn model
%
%   Keys that can be passed in:
%
%   start             Index of the first image in imdb to process
%   end               Index of the last image in imdb to process
%   crop_mode         Crop mode (either 'warp' or 'square')
%   crop_padding      Amount of padding in crop
%   net_file          Path to the Caffe CNN to use
%   cache_name        Path to the precomputed feature cache

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addParamValue('layer', 7, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);
ip.addOptional('net_file', ...
    './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', ...
    @isstr);
ip.addOptional('cache_name', ...
    'v1_finetune_voc_2007_trainval_iter_70000', @isstr);

ip.parse(imdb, varargin{:});
opts = ip.Results;
%opts.net_def_file = './model-defs/cub_finetune_deploy_fc7.prototxt';

image_ids = imdb.image_ids;
if opts.end == 0
    opts.end = length(image_ids);
end

% Where to save feature cache
opts.output_dir = ['./feat_cache/' opts.cache_name '/' imdb.name '/'];
mkdir_if_missing(opts.output_dir);

% Log feature extraction
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [opts.output_dir 'rcnn_compute_scores_' timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% load the region of interest database
regions_file = sprintf('./data/selective_search_data/%s', imdb.name);
try
    regions = load(regions_file);
catch
    disp('Run selective search on cub2011 images first.');
    exit(-1);
end

opts.net_def_file = './model-defs/rcnn_batch_256_output_fc7.prototxt';

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;

load(sprintf('cachedir/parts_train/feature_stats_parts_train_layer_%d_%s.mat', ...
                    opts.layer, opts.cache_name));

opts.feat_norm_mean = mean_norm;

%outFName = sprintf('feat_cache/%s/parts_test/rcnn_scores_%dto%d.mat', opts.cache_name, opts.start, opts.end);
%scores = cell(opts.end-opts.start+1,1);

total_time = 0;
count = 0;
for i = opts.start:opts.end
    fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);
    
    save_file = [opts.output_dir image_ids{i} '_score.mat'];
    [tmp1,~,~]=fileparts(save_file);
    mkdir_if_missing(fullfile(tmp1));
    if exist(save_file, 'file') ~= 0
        fprintf(' [already exists]\n');
        continue;
    end
    
    pool5_file = [opts.output_dir image_ids{i} '.mat'];
    [tmp1,~,~]=fileparts(pool5_file);
    mkdir_if_missing(fullfile(tmp1));
    count = count + 1;
    tot_th = tic;
    
    if exist(pool5_file, 'file') ~= 0
        fprintf(' [pool 5 already exists]\n');
        th = tic;
        d = load(pool5_file);
        d.feat = rcnn_pool5_to_fcX(d.feat, opts.layer, rcnn_model);
        fprintf(' [pool5 to fc7 features: %.3fs]\n', toc(th));
    else
        boxes = regions.boxes{i};
        d.boxes = boxes(:, [2 1 4 3]);
        im = imread(imdb.image_at(i));
        th = tic;
        d.feat = rcnn_features(im, d.boxes, rcnn_model);
        fprintf(' [fc7 features: %.3fs]\n', toc(th));
    end
    
    d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
    zs = bsxfun(@plus, d.feat*detector_model.W, detector_model.B);
    %scores{i} = zs;
    %if mod(i,10)==0
    %    disp('Save scores...');
    %    save(outFName, 'scores');
    %end
    
    d.score = zs;
    d = rmfield(d,'feat');
    th = tic;
    save(save_file, '-struct', 'd');
    fprintf(' [saving:   %.3fs]\n', toc(th));
    
    total_time = total_time + toc(tot_th);
    fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
            total_time/count, total_time);
    
    
end
