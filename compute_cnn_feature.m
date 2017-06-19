function d = compute_cnn_feature(config, images, boxes)
% compute rcnn fetaure
opts.net_file     = 'caches/CUB_bbox_finetune.caffe_model';
opts.cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
opts.crop_mode    = 'warp';
opts.net_def_file = 'caches/cub_finetune_deploy_fc7.prototxt';
opts.output_dir = 'caches_rcnn/';
opts.crop_padding = 16;

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;

nim = length(images);

total_time = 0;
count = 0;
for i = 1:nim
  fprintf('%s: cache features: %d/%d\n', procid(), i, nim);

  save_file = [opts.output_dir images{i} '.mat'];
  [tmp1,~,~]=fileparts(save_file);
  mkdir_if_missing(fullfile(tmp1));
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;

  tot_th = tic;

  im = imread(fullfile(config.img_base, images{i}));

  th = tic;
  box = boxes{i};
  box = box(:, [2 1 4 3]);
  d.feat = rcnn_features(im, box, rcnn_model);
  d.boxes = boxes{i};
  fprintf(' [features: %.3fs]\n', toc(th));

  th = tic;
  save(save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs]\n', toc(th));

  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);
end



    
end