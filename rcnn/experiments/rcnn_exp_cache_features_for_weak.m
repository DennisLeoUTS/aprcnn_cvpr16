function rcnn_exp_cache_features_for_weak(imdb, part)
% chunk: train, test
% part: bbox, head, body


% -------------------- CONFIG --------------------
net_file     = sprintf('/home/zxu/workspace/caffe/examples/cub-finetuning/cub_finetune_%s_iter_10000.caffemodel', part);
cache_name   = sprintf('v1_finetune_cub_train_%s_iter_10k', part);
crop_mode    = 'warp';
crop_padding = 16;
% ------------------------------------------------

rcnn_cache_pool5_features_for_weak(imdb, ...
    'crop_mode', crop_mode, ...
    'crop_padding', crop_padding, ...
    'net_file', net_file, ...
    'cache_name', cache_name);
end

