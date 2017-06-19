%function rcnn_run

%% rcnn
% go to ./rcnn directory

%% startup
global RCNN_CONFIG_OVERRIDE;
conf_override.prcnn_path = '..';
eval(sprintf('run %s/config_prcnn()', conf_override.prcnn_path));
conf_override.dir = ans.dir;                 % get global paths
conf_override.config = ans.config;           % get part-based rcnn settings
RCNN_CONFIG_OVERRIDE = @() conf_override;
startup;

conf = rcnn_config;

%% load imbd and extract selective search
imdb_train = imdb_from_voc('train');
extract_ss_on_cub( imdb_train );
imdb_test = imdb_from_voc('test');
extract_ss_on_cub( imdb_test );

% on weak
imdb_weak = imdb_from_voc('weak');
extract_ss_on_cub( imdb_weak );

%% finetune cnn
partbased_rcnn_make_window_file(imdb_train, conf.dir.CAFFE_FINETUNE_PATH, conf.config);
partbased_rcnn_make_window_file(imdb_test, conf.dir.CAFFE_FINETUNE_PATH, conf.config);
% finetune cnn in shell mode, see run_shell.sh in caffe_finetune_path

%% extract cnn features
chunks = {'train', 'test', 'weak'};
parts = conf.config.parts;
for i=1:length(chunks)
    for j=1:length(parts)
        rcnn_exp_cache_features(chunks{i}, parts{j});
    end
end

%% train rcnn models
for i=1:length(parts)
    rcnn_exp_train_and_test(parts{i});
end

%% test rcnn models
for i=1:length(parts)
    % compute rcnn detector scores, #!! different parts should be implemented in different matlab consoles
    rcnn_exp_test_scores(parts{i});
end


%% part-based rcnn
% go to .directory
% get part-based rcnn results before re-finetuning
eval(sprintf('run %s/init', conf.prcnn_path));
cmd = sprintf('run %s/run_prcnn(1,10000)', conf.prcnn_path);
eval(cmd);
res_before = ans;

%% re-finetune cnn
selected_samples = load(conf.dir.SELECTED_WEAK_PATCH_FILE);
selected_samples = selected_samples.selected;
eval(sprintf('run %s/config_prcnn(2,20000)', conf_override.prcnn_path));
conf_override.dir = ans.dir;
conf_override.config = ans.config;
RCNN_CONFIG_OVERRIDE = @() conf_override;
conf = rcnn_config;

% if want to use all weak images to finetune cnns
%for j=1:3
%    selected_samples{j}=1:length(imdb_weak.image_ids);
%end
partbased_rcnn_make_window_file_weak(imdb_train, imdb_weak, selected_samples, conf.dir.CAFFE_FINETUNE_PATH);
partbased_rcnn_make_window_file(imdb_test, conf.dir.CAFFE_FINETUNE_PATH);
% finetune cnn in shell mode, see run_shell.sh in caffe_finetune_path


%% final prcnn result
cmd = sprintf('run %s/run_prcnn(2,20000)', conf.prcnn_path);
eval(cmd);
res_after = ans;
%end
