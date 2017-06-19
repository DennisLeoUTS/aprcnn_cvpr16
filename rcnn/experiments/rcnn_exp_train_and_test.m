function rcnn_exp_train_and_test(part)
% Runs an experiment that trains an R-CNN model and tests it.

% -------------------- CONFIG --------------------
conf = rcnn_config;
net_file     = sprintf(conf.dir.CAFFE_NET_FILE, part);
cache_name   = sprintf(conf.dir.RCNN_FEATURE_PATH, part);
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
%layer         = 6;
k_folds      = 0;

% ------------------------------------------------

imdb_train = imdb_from_voc('train');
%imdb_test = imdb_from_voc('test');

[rcnn_model, rcnn_k_fold_model] = ...
    rcnn_train_4class(imdb_train, ...
      'layer',        layer, ...
      'k_folds',      k_folds, ...
      'cache_name',   cache_name, ...
      'net_file',     net_file, ...
      'crop_mode',    crop_mode, ...
      'crop_padding', crop_padding,...
      'part_name', part, ...
      'checkpoint', 500);

%if k_folds > 0
%  res_train = rcnn_test(rcnn_k_fold_model, imdb_train);
%else
%  res_train = [];
%end
%res_test = rcnn_test(rcnn_model, imdb_test);
