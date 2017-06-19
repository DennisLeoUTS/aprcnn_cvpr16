function imdb = imdb_from_voc(image_set)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the CUB-200 dataset.
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
% Changed by Zhe Xu based on RCNN repo

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

conf = rcnn_config;

if strcmp(image_set, 'train')
    cache_file = conf.dir.IMDB_TRAIN_FILE;
elseif strcmp(image_set, 'test')
    cache_file = conf.dir.IMDB_TEST_FILE;
elseif strcmp(image_set, 'weak');
    imdb = imdb_from_weak;
    return;
else
    error('Image set name not valid!');
end

try
  load(cache_file);
catch
  assert(strcmp(image_set, 'train') || strcmp(image_set, 'test'));
  imdb.name = ['parts' '_' image_set];
  BIRD_DIR = conf.dir.BIRD_DIR;  
  imdb.image_dir = fullfile(BIRD_DIR, 'images/'); % change to your path
  imdir = fullfile(BIRD_DIR,  'images.txt');
  [img_id img_path] = textread(imdir,'%d %s');  
  traintest_dir = [BIRD_DIR 'train_test_split.txt'];
  [img_id train_flag] = textread(traintest_dir, '%d %d');
  if strcmp(image_set, 'train')
      idx = find(train_flag == 1);
  else
      idx = find(train_flag == 0);
  end
  imdb.image_ids = img_path(idx);
  classdir = fullfile(BIRD_DIR, 'classes.txt');
  [class_id classes] = textread(classdir, '%d %s');
  imdb.parts = {'bbox', 'head', 'body'};
  imdb.num_parts = length(imdb.parts);
  imdb.classes = classes;
  imdb.num_classes = length(imdb.classes);
  imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
  imdb.class_ids = 1:imdb.num_classes;
  imdb.part_to_id = ...
    containers.Map(imdb.parts, 1:imdb.num_parts);

  % VOC specific functions for evaluation and region of interest DB
  imdb.eval_func = @imdb_eval_voc;
  imdb.roidb_func = @roidb_from_voc;
  imdb.image_at = @(i) ...
      sprintf('%s/%s', imdb.image_dir, imdb.image_ids{i});

  for i = 1:length(imdb.image_ids)
    tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
    info = imfinfo(sprintf('%s/%s', imdb.image_dir, imdb.image_ids{i}));
    imdb.sizes(i, :) = [info.Height info.Width];
  end

  fprintf('Saving imdb to cache...');
  save(cache_file, 'imdb', '-v7.3');
  fprintf('done\n');
  end
end
