function rcnn_exp_test_scores_for_weak(part)
% compute scores using rcnn model for testing images
% part: bbox, head, body


% -------------------- CONFIG --------------------
net_file     = sprintf('/home/zxu/workspace/caffe/examples/cub-finetuning/cub_finetune_%s_iter_10000.caffemodel', part);
cache_name   = sprintf('v1_finetune_cub_train_%s_iter_10k', part);
crop_mode    = 'warp';
crop_padding = 16;
rcnn_model   = sprintf('./cachedir/parts_train/rcnn_model_%s.mat', part);
load(rcnn_model);
layer        = 7;
% ------------------------------------------------

load('../imdb_weak.mat');

rcnn_compute_score_for_weak(imdb, rcnn_model.detectors, ...
    'crop_mode', crop_mode, ...
    'crop_padding', crop_padding, ...
    'cache_name', cache_name, ...
    'net_file', net_file, ...
    'cache_name', cache_name,...
    'layer', layer);
end

