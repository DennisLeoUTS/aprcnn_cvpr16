function startup

addpath('selective_search');
if exist('selective_search/SelectiveSearchCodeIJCV')
    addpath('selective_search/SelectiveSearchCodeIJCV');
    addpath('selective_search/SelectiveSearchCodeIJCV/Dependencies');
else
    fprintf('Warning: you will need the selective search IJCV code.\n');
    fprintf('Press any key to download it (runs ./selective_search/fetch_selective_search.sh)> ');
    pause;
    system('/selective_search/fetch_selective_search.sh');
    addpath('selective_search/SelectiveSearchCodeIJCV');
    addpath('selective_search/SelectiveSearchCodeIJCV/Dependencies');
end
addpath('vis');
addpath('utils');
addpath('bin');
addpath('nms');
addpath('finetuning');
addpath('bbox_regression');

conf = rcnn_config;

if exist(conf.dir.CAFFE_MATLAB_PATH)
    addpath(conf.dir.CAFFE_MATLAB_PATH);
else
    warning('Please install Caffe in ./external/caffe');
end
addpath('experiments');
addpath('imdb');
addpath('vis/pool5-explorer');
addpath('examples');
fprintf('R-CNN startup done\n');

end