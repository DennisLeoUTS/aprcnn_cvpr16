%% Run classification given pretrained cnn models and model definition files.
%% Written by Ning Zhang

function results = run_classification(MODE, ITER)
% MODE: 1: training, 2: weak
% ITER: # of iterations used in finetuning caffe models

% get data
prcnn = config_prcnn(MODE, ITER);

%% Extract features from train box and then train linear SVM.
try
    load(prcnn.dir.TRAIN_MODEL_FILE);
catch
    try
        load(prcnn.dir.TRAIN_FEA_FILE);
    catch
        train_fea = cell(1, prcnn.config.N_parts);
        for i = 1 : prcnn.config.N_parts  % use 3 different matlab terminals
            train_fea{i} = extract_deep_feature(prcnn.dir.CNN_MODELS{i},...
                prcnn.dir.MODEL_DEF_FILE, prcnn.config.impathtrain, prcnn.config.train_box{i});
        end
        save(prcnn.dir.TRAIN_FEA_FILE, 'train_fea', '-v7.3');
    end
    
    TRN_fea = [];
    for i = 1 : prcnn.config.N_parts
        TRN_fea = [TRN_fea train_fea{i}];
    end
    TRN_fea = scale_feature(TRN_fea);
    
    %% weak
    if MODE==2
        % compute weak features
        tmp = load(prcnn.dir.IMDB_WEAK_FILE);
        imdb_weak = tmp.imdb;
        try 
            load(prcnn.dir.WEAK_FEA_FILE);
        catch
            Nim = length(imdb_weak.image_ids);
            if exist(prcnn.dir.WEAK_DETECT_BOX_FILE, 'file')
                load(prcnn.dir.WEAK_DETECT_BOX_FILE);
            else
                detect_boxes = test_rcnn_parts_for_weak(prcnn);
            end
            weak_fea = cell(prcnn.config.N_parts, 1);
            imweakpath = arrayfun(@(i)fullfile(imdb_weak.image_dir, imdb_weak.image_ids{i}),...
                1:Nim, 'UniformOutput', false);
            
            for i = 1 : prcnn.config.N_parts
                weak_fea{i} = extract_deep_feature(prcnn.dir.CNN_MODELS{i}, ...
                    prcnn.dir.MODEL_DEF_FILE, imweakpath, detect_boxes{i, prcnn.para.selected_geo_method});
            end
            save(prcnn.dir.WEAK_FEA_FILE, 'weak_fea', '-v7.3');
        end
        WEAK_fea = [];
        for i = 1 : prcnn.config.N_parts
            WEAK_fea = [WEAK_fea weak_fea{i}];
        end
        WEAK_fea = scale_feature(WEAK_fea);
    end
    
    disp('Train linear SVM ... ...');
    model = train(prcnn.config.trainlabel, sparse(double(TRN_fea)), prcnn.para.svm_option);
    save(prcnn.dir.TRAIN_MODEL_FILE, 'model');

end

%% Extract features from detected box and then test.
try
    load(prcnn.dir.TEST_FEA_FILE);
catch
    % get detected part boxes
    detect_boxes = get_rcnn_detections(prcnn);
    
    % test_fea{i,j} is the features for part i of method j
    test_fea = cell(prcnn.config.N_parts, prcnn.config.N_methods);
    for i = 1 : prcnn.config.N_parts     % use 3 different matlab terminals
        for method = 1 : prcnn.config.N_methods
            % method = 1 box detection feature
            % method = 2 prior detection feature
            test_fea{i, method} = extract_deep_feature(prcnn.dir.CNN_MODELS{i}, ...
                prcnn.dir.MODEL_DEF_FILE, prcnn.config.impathtest, detect_boxes{i, method});
        end
    end
    % features for groundtruth bounding box
    test_fea_gt = cell(1, prcnn.config.N_parts);
    for i = 1 : prcnn.config.N_parts    % use 3 different matlab terminals
        test_fea_gt{i} = extract_deep_feature(prcnn.dir.CNN_MODELS{i}, ...
            prcnn.dir.MODEL_DEF_FILE, prcnn.config.impathtest, prcnn.config.test_box{i});
    end
    save(prcnn.dir.TEST_FEA_FILE,'test_fea', 'test_fea_gt');
end

%% Test SVM model
TST_fea = [];
for i = 1 : prcnn.config.N_parts
    TST_fea = [TST_fea test_fea_gt{i}];
end
TST_fea = scale_feature(TST_fea);
[~,accuracy,~] = predict(prcnn.config.testlabel, sparse(double(TST_fea)), model);
results.oracle_accuracy = accuracy(1);
fprintf('Accuracy of using oracle part boxes is %f\n', accuracy(1));

for method = 1 : prcnn.config.N_methods
    TST_fea = [];
    for i = 1 : prcnn.config.N_parts
        TST_fea = [TST_fea test_fea{i, method}];
    end
    TST_fea = scale_feature(TST_fea);
    [pd,accuracy,~] = predict(prcnn.config.testlabel, sparse(double(TST_fea)), model);
    accuracy = accuracy(1);
    results.detected_accuracy(method) = accuracy;
    fprintf('Accuracy of %s is %f\n', prcnn.config.methods{method}, accuracy);
end


end

function fea = scale_feature(fea)
ppp = 0.3;
for i = 1:size(fea,2)
    fea(:,i) = sign(fea(:,i)).*abs(fea(:,i)).^ppp;
end
end
