function rcnn_exp_cache_features(chunk)

% -------------------- CONFIG --------------------
net_file     = '/home/zxu/workspace/part-based-RCNN-master/caches/CUB_head_finetune.caffe_model';
cache_name   = 'v1_finetune_cub_train_given_head';
crop_mode    = 'warp';
crop_padding = 16;
% ------------------------------------------------

imdb_train = imdb_from_voc('', 'train', '');
%imdb_val   = imdb_from_voc(VOCdevkit, 'val', '2007');
imdb_test  = imdb_from_voc('', 'test', '');
%imdb_trainval = imdb_from_voc(VOCdevkit, 'trainval', '2007');

switch chunk
  case 'train'
    rcnn_cache_pool5_features(imdb_train, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
  case 'test'
    end_at = length(imdb_test.image_ids);
    rcnn_cache_pool5_features(imdb_test, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'cache_name', cache_name);
end


% ------------------------------------------------------------------------
function link_up_trainval(cache_name, imdb_split, imdb_trainval)
% ------------------------------------------------------------------------
cmd = {['mkdir -p ./feat_cache/' cache_name '/' imdb_trainval.name '; '], ...
    ['cd ./feat_cache/' cache_name '/' imdb_trainval.name '/; '], ...
    ['for i in `ls -1 ../' imdb_split.name '`; '], ... 
    ['do ln -s ../' imdb_split.name '/$i $i; '], ... 
    ['done;']};
cmd = [cmd{:}];
fprintf('running:\n%s\n', cmd);
system(cmd);
fprintf('done\n');
