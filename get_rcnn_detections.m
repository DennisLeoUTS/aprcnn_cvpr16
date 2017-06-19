%% function to get rcnn detected part boxes
%% Written by Ning Zhang

function detect_boxes = get_rcnn_detections(prcnn)
try 
  load(prcnn.dir.DETECT_BOX_FILE);
catch
  detect_boxes = test_rcnn_parts_by_xz(prcnn);
  save(prcnn, 'detect_boxes');
end
end

