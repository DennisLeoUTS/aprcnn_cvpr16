function roidb = roidb_from_voc(imdb)
% roidb = roidb_from_voc(imdb)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% Change by Zhe Xu
conf = rcnn_config;

if strcmp(imdb.name, 'parts_train')
    cache_file = conf.dir.ROIDB_TRAIN_FILE;
else
    cache_file = conf.dir.ROIDB_TEST_FILE;
end

try
  load(cache_file);
catch
  roidb.name = imdb.name;

  fprintf('Loading region proposals...');
  
  regions_file = sprintf(conf.dir.SELECTIVE_SEARCH_FILE, roidb.name);
  % follow the selective search instructions and run selective search on bird images.
  try 
    regions = load(regions_file);
  catch
    disp('Run selective search on cub2011 images first.');
    exit(-1);
  end
  fprintf('done\n');

  
  config = conf.config;
  %nvalid_bbox = zeros(floor(length(imdb.image_ids)/30), 2);
  %n1=0;
  %n2=0;
  for i = 1:length(imdb.image_ids)
    tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
    voc_rec = get_pascal_format(config, imdb.image_ids{i});
%     if strcmp(imdb.name, 'parts_train')
%         gtlabel = config.trainlabel(i);
%     else
%         gtlabel = config.testlabel(i);
%     end
    part_to_id = containers.Map({'bbox','head','body'}, 1:3);
    roidb.rois(i) = attach_proposals(voc_rec, regions.boxes{i}, part_to_id);
%     yy=roidb.rois(i).overlap(4:end,:);
%     xx=max(yy);
%     nhead = sum(yy(:,2)>.5);
%     nbody = sum(yy(:,3)>.5);
%     n1 = n1+nhead;
%     n2 = n2+nbody;
%     %fprintf('# ss: %d, maxoverlap: %f, %f, %f, valid nbox: %d, %d\n',...
%     %    size(yy,1), xx(1), xx(2), xx(3), nhead, nbody);
%     if mod(i,30)==0
%         %fprintf('Class %d: valid bbox for head: %d, body: %d\n', i/30, n1, n2);
%         nvalid_bbox(i/30,:)=[n1, n2];
%         n1=0;
%         n2=0;
%     end
  end

  fprintf('Saving roidb to cache...');
  save(cache_file, 'roidb', '-v7.3');
  fprintf('done\n');
end
end





function voc_rec = get_pascal_format(config, img_path)
  % find image id for img_path
  full_path = [config.img_base img_path];
  train_idx = find(ismember(config.impathtrain, full_path));
  test_idx = find(ismember(config.impathtest, full_path));
  assert(isempty(train_idx) || isempty(test_idx));
  assert(~isempty(train_idx) || ~isempty(test_idx));
  if ~isempty(train_idx)
    voc_rec.objects(1).bbox = config.train_box{1}(train_idx,:);
    voc_rec.objects(1).class = 'bbox';
    count = 2;
    if config.train_box{2}(train_idx,1) ~= -1
      voc_rec.objects(count).bbox = config.train_box{2}(train_idx,:);
      voc_rec.objects(count).class = 'head';
      count = count + 1;
    end
    if config.train_box{3}(train_idx,1) ~= -1
      voc_rec.objects(count).bbox = config.train_box{3}(train_idx,:);
      voc_rec.objects(count).class = 'body';
    end
  end
  if ~isempty(test_idx)
    voc_rec.objects(1).bbox = config.test_box{1}(test_idx,:);
    voc_rec.objects(1).class = 'bbox';
    count = 2;
    if config.test_box{2}(test_idx,1) ~= -1
      voc_rec.objects(count).bbox = config.test_box{2}(test_idx,:);
      voc_rec.objects(count).class = 'head';
      count = count + 1;
    end
    if config.test_box{3}(test_idx,1) ~= -1
      voc_rec.objects(count).bbox = config.test_box{3}(test_idx,:);
      voc_rec.objects(count).class = 'body';
    end
  end 
end 


% ------------------------------------------------------------------------
function rec = attach_proposals(voc_rec, boxes, class_to_id)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]
boxes = boxes(:, [2 1 4 3]);

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]
if isfield(voc_rec, 'objects')
  gt_boxes = cat(1, voc_rec.objects(:).bbox);
  all_boxes = cat(1, gt_boxes, boxes);
  gt_classes = class_to_id.values({voc_rec.objects(:).class});
  gt_classes = cat(1, gt_classes{:});
  %gt_classes = gtlabel;
  num_gt_boxes = size(gt_boxes, 1);
else
  gt_boxes = [];
  all_boxes = boxes;
  gt_classes = [];
  num_gt_boxes = 0;
end
num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
end
