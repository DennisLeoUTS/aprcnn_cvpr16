function partbased_rcnn_make_window_file(imdb, out_dir, config)
% rcnn_make_window_file(imdb, out_dir)
%   Makes a window file that can be used by the caffe WindowDataLayer
%   for finetuning.
%
%   The window file format contains repeated blocks of:
%
%     # image_index
%     img_path
%     channels
%     height
%     width
%     num_windows
%     class_index overlap x1 y1 x2 y2
%     <... num_windows-1 more windows follow ...>

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
%
% This file is part of the R-CNN code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

roidb = imdb.roidb_func(imdb);

window_file_bbox = sprintf('%s/window_file_%s_bbox.txt', ...
    out_dir, imdb.name);
window_file_head = sprintf('%s/window_file_%s_head.txt', ...
    out_dir, imdb.name);
window_file_body = sprintf('%s/window_file_%s_body.txt', ...
    out_dir, imdb.name);
fid(1) = fopen(window_file_bbox, 'wt');
fid(2) = fopen(window_file_head, 'wt');
fid(3) = fopen(window_file_body, 'wt');

channels = 3; % three channel images

for i = 1:length(imdb.image_ids)
    tic_toc_print('make window file: %d/%d\n', i, length(imdb.image_ids));
    if strcmp(imdb.name, 'parts_train')
        label = config.trainlabel(i);
    else
        label = config.testlabel(i);
    end
    img_path = imdb.image_at(i);
    roi = roidb.rois(i);
    num_boxes = size(roi.boxes, 1);
    for p=1:3
        fprintf(fid(p), '# %d\n', i-1);
        fprintf(fid(p), '%s\n', img_path);
        fprintf(fid(p), '%d\n%d\n%d\n', ...
            channels, ...
            imdb.sizes(i, 1), ...
            imdb.sizes(i, 2));
        fprintf(fid(p), '%d\n', num_boxes);
    end
    for j = 1:num_boxes
        for p = 1:3
            ov = roi.overlap(j,p);
            % zero overlap => label = 0 (background)
            label_tmp = label;
            if ov < 1e-5
                label_tmp = 0;
                ov = 0;
            end
            bbox = roi.boxes(j,:)-1;
            fprintf(fid(p), '%d %.3f %d %d %d %d\n', ...
                label_tmp, ov, bbox(1), bbox(2), bbox(3), bbox(4));
        end
    end
end

for i=1:3
    fclose(fid(i));
end
