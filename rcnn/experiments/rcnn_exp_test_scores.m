function rcnn_exp_test_scores(part)
% compute scores using rcnn model for testing images
% part: bbox, head, body


% -------------------- CONFIG --------------------
conf = rcnn_config;
net_file     = sprintf(conf.dir.CAFFE_NET_FILE, part);
cache_name   = sprintf(conf.dir.RCNN_FEATURE_PATH, part);
crop_mode    = 'warp';
crop_padding = 16;
rcnn_model   = sprintf(conf.dir.RCNN_DETECT_MODEL_FILE, 'parts_train', part);
load(rcnn_model);
layer        = 7;
% ------------------------------------------------

imdb_test  = imdb_from_voc('test');

rcnn_compute_score(imdb_test, rcnn_model.detectors, ...
    'crop_mode', crop_mode, ...
    'crop_padding', crop_padding, ...
    'cache_name', cache_name, ...
    'net_file', net_file, ...
    'cache_name', cache_name,...
    'layer', layer);
end

