function get_detection_results(imdb, detect_boxes)
% get rcnn detection results: iou
% method: two part-based rcnn geometric methods: box and MG
% part: bbox, head, body
% example: 
% --  imdb_test = imdb_from_voc('','test','');
% --  load('../caches/rcnn_detect_boxes_v2.mat');
% --  get_detection_results(imdb_test, detect_boxes);

pcp = cell(2,3);
for i=1:6, pcp{i}=[]; end

roidb = imdb.roidb_func(imdb);
for i=1:length(imdb.image_ids)
    roi = roidb.rois(i);
    Nbox = size(roi.boxes,1);
    for method = 1:2
        for p=1:3
            tmp = abs(repmat(detect_boxes{p,method}(i,1:4),Nbox,1)-roi.boxes);
            gt = find(sum(tmp,2)==0);
            ov = roi.overlap(gt,p);
            pcp{method,p} = cat(1,pcp{method,p},ov>=.5);
        end
    end
end

pcp_results = arrayfun(@(i)mean(pcp{i}), 1:6);
pcp_results = reshape(pcp_results, 2,3)

end