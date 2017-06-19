% Run classification given pretrained cnn models and model definition files.
% introducing weakly-labeled data to improve performance

function results = run_classification_weak(cnn_models, model_def)
% get data
try
    load('caches/cub2011_config.mat');
catch
    config = get_bird_data;
    save('caches/cub2011_config.mat', 'config');
end

% Extract features from train box and then train linear SVM.
try
    load('caches/finetune_model_fc7_after_weak.mat');
catch
    try
        load('caches/finetune_train_fea_fc7.mat');
    catch
        train_fea = cell(1, config.N_parts);
        for i = 1 : config.N_parts  % use 3 different matlab terminals
            train_fea{i} = extract_deep_feature(cnn_models{i}, model_def, ...
                config.impathtrain, config.train_box{i});
        end
        save('caches/finetune_train_fea_fc7.mat', 'train_fea', '-v7.3');
    end
    
    try
        load('caches/finetune_weak_fea_fc7.mat');
    catch
        tmp = load('imdb_weak.mat');
        imdb_weak = tmp.imdb;
        Nim = length(imdb_weak.image_ids);
        if exist('caches/rcnn_detect_boxes_weak.mat', 'file')
            load('caches/rcnn_detect_boxes_weak.mat');
        else
            detect_boxes = test_rcnn_parts_for_weak_v2(config);
        end
        weak_fea = cell(config.N_parts, 1);
        imweakpath = arrayfun(@(i)fullfile(imdb_weak.image_dir, imdb_weak.image_ids{i}),...
            1:Nim, 'UniformOutput', false);
        
        scores = zeros(Nim, config.N_parts);
        for i = 1:Nim
            for j=1:config.N_parts
                scores(i,j) = detect_boxes{j,2}(i,end);
            end
        end
        ss = scores(:,1).*scores(:,2).*scores(:,3);
        %[ss,tt]=sort(ss,'descend');
        valid  = ss>.2;
        
        for i = 1 : config.N_parts
            weak_fea{i} = extract_deep_feature(cnn_models{i}, ...
                model_def, imweakpath, detect_boxes{i, 2});
        end
        save('caches/finetune_weak_fea_fc7.mat', 'weak_fea', 'valid', 'scores', '-v7.3');
    end
    
    TRN_fea = [];
    for i = 1 : config.N_parts
        TRN_fea = [TRN_fea train_fea{i}];
    end
    WEAK_fea = [];
    for i = 1 : config.N_parts
        WEAK_fea = [WEAK_fea weak_fea{i}(find(valid),:)];
    end
    TRN_fea_all = [TRN_fea; WEAK_fea];
    TRN_fea = scale_feature(TRN_fea);
    TRN_fea_all = scale_feature(TRN_fea_all);
    disp('Train linear SVM ... ...');
    lc = 1; % regularization parameter C
    option = ['-s 1 -c ' num2str(lc)];
    model = train(config.trainlabel, sparse(double(TRN_fea)), option);
    
    trainlabel_weak = imdb_weak.labels;
    trainlabel_all = [config.trainlabel; trainlabel_weak(valid)];
    model_weak = train(trainlabel_all, sparse(double(TRN_fea_all)), option);
    save('caches/finetune_model_fc7_after_weak', 'model', 'model_weak');
end

% Extract features from detected box and then test.
try
    load('caches/finetune_test_fea_fc7.mat');
catch
    % get detected part boxes
    detect_boxes = get_rcnn_detections(config);
    
    % test_fea{i,j} is the features for part i of method j
    test_fea = cell(config.N_parts, config.N_methods);
    for i = 1 : config.N_parts     % use 3 different matlab terminals
        for method = 1 : config.N_methods
            % method = 1 box detection feature
            % method = 2 prior detection feature
            test_fea{i, method} = extract_deep_feature(cnn_models{i}, ...
                model_def, config.impathtest, detect_boxes{i, method});
        end
    end
    % features for groundtruth bounding box
    test_fea_gt = cell(1, config.N_parts);
    for i = 1 : config.N_parts    % use 3 different matlab terminals
        test_fea_gt{i} = extract_deep_feature(cnn_models{i}, ...
            model_def, config.impathtest, config.test_box{i});
    end
    save('caches/finetune_test_fea_fc7','test_fea', 'test_fea_gt');
end

% Test SVM model
TST_fea = [];
for i = 1 : config.N_parts
    TST_fea = [TST_fea test_fea_gt{i}];
end
TST_fea = scale_feature(TST_fea);
[~,accuracy,~] = predict(config.testlabel, sparse(double(TST_fea)), model);
results.oracle_accuracy = accuracy(1);
fprintf('Accuracy of using oracle part boxes is %f\n', accuracy);

for method = 1 : config.N_methods
    TST_fea = [];
    for i = 1 : config.N_parts
        TST_fea = [TST_fea test_fea{i, method}];
    end
    TST_fea = scale_feature(TST_fea);
    [~,accuracy,~] = predict(config.testlabel, sparse(double(TST_fea)), model);
    [~,accuracy2,~] = predict(config.testlabel, sparse(double(TST_fea)), model_weak);
    accuracy = accuracy(1);
    results.detected_accuracy(method) = accuracy;
    fprintf('Accuracy of %s is %f\n', config.methods{method}, accuracy);
end
end

function fea = scale_feature(fea)
ppp = 0.3;
for i = 1:size(fea,2)
    fea(:,i) = sign(fea(:,i)).*abs(fea(:,i)).^ppp;
end
end
