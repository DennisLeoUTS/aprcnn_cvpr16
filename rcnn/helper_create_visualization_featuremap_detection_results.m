% following create_visualization_for_paper_featuremap_detection_results

%% init
%{=
global RCNN_CONFIG_OVERRIDE;
conf_override.prcnn_path = '..';
eval(sprintf('run %s/config_prcnn()', conf_override.prcnn_path));
conf_override.dir = ans.dir;                 % get global paths
conf_override.config = ans.config;           % get part-based rcnn settings
RCNN_CONFIG_OVERRIDE = @() conf_override;
startup;
conf = rcnn_config;

imdb = imdb_from_voc('test');
roidb = roidb_from_voc(imdb);

cache_name   = sprintf(conf.dir.RCNN_FEATURE_PATH, part);
[~,cache_name,~] = fileparts(cache_name);
load(sprintf(conf.dir.RCNN_FEATURE_STAT_FILE, ...
                    'parts_train', 'parts_train', 7, cache_name));
%}
%%
for partID = 1:3
    part = prcnn.config.parts{partID};
    %net_file     = sprintf(conf.dir.CAFFE_NET_FILE, part);
    net_file = sprintf('%s/examples/cub-finetuning-weak/cub_finetune_%s_iter_20000.caffemodel', conf.dir.CAFFE_PATH, part);
    net_def_file = './model-defs/rcnn_batch_256_output_fc7.prototxt';
    rcnn_model = rcnn_create_model(net_def_file, net_file);
    rcnn_model = rcnn_load_model(rcnn_model);
    
    rcnn_model_name   = sprintf(conf.dir.RCNN_DETECT_MODEL_FILE, 'parts_train', part);
    tmp = load(rcnn_model_name);
    detector_model = tmp.rcnn_model.detectors;
    
    %%
    %R = randperm(Nsample);
    %R = 1:Nsample;
    R = [20 26 543 564 689 1914 2137 2243 5700 4416 4470 3896 2363 2367];
    Nsample = length(R);
    for j=1:Nsample
        try
        %i = inds(R(j));
        i = R(j);
        l = prcnn.config.testlabel(i);
        pool5_file = sprintf(prcnn.dir.RCNN_POOL5_FILE, imdb.name, imdb.image_ids{i});
        
        d = roidb.rois(i);
        im = imread(imdb.image_at(i));
        th = tic;
        
        feat = rcnn_features(im, d.boxes, rcnn_model);
        feat = rcnn_scale_features(feat, mean_norm);
        %res = feat*model.w(l,(partID-1)*4096+1:partID*4096)';
        res = bsxfun(@plus, feat*detector_model.W, detector_model.B);
        
        fprintf(' [fc7 features: %.3fs]\n', toc(th));
        
        %figure(1),imshow(im);
        res1 = res;
        fmap = zeros(size(im,1),size(im,2));
        fmap_div = fmap;
        for t=1:length(res)
            bb=d.boxes(t,:);
            fmap(bb(2):bb(4),bb(1):bb(3)) = fmap(bb(2):bb(4),bb(1):bb(3))+res1(t);
            fmap_div(bb(2):bb(4),bb(1):bb(3)) = fmap_div(bb(2):bb(4),bb(1):bb(3))+1;
        end
        me = mean(fmap_div(:));
        %figure(2),imshow(fmap,[],'Border','Tight');
        %colormap('jet');
        h=figure(5);
        clf,imshow(fmap./fmap_div,[],'Border','Tight');
        colormap('jet');
        
        outFName = sprintf('../results/figure_nn/%04d/%d_detect_fmap.jpg', i, partID);
        [yy,~,~]=fileparts(outFName);
        if ~exist(yy,'dir')
            mkdir(yy);
        end
        print(h, '-djpeg', outFName);
        catch
            fprintf('Something wrong for %d\n', i);
            continue;
        end
    end
end
