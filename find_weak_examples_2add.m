function acc_test = find_weak_examples_2add(MODE, ITER)
% given part-based rcnn models, find images in the weakly-supervised
% dataset which are proper to be inserted into the training set

% config
prcnn = config_prcnn(MODE, ITER);

%% detect on weak dataset
try
    load(prcnn.dir.WEAK_FEA_FILE);
catch
    tmp = load(prcnn.dir.IMDB_WEAK_FILE);
    imdb_weak = tmp.imdb;
    Nim = length(imdb_weak.image_ids);
    if exist(prcnn.dir.WEAK_DETECT_BOX_FILE, 'file')
        load(prcnn.dir.WEAK_DETECT_BOX_FILE);
    else
        detect_boxes = test_rcnn_parts_for_weak(prcnn);
    end
    weak_fea = cell(prcnn.config.N_parts, 1);
    imweakpath = arrayfun(@(i)fullfile(imdb_weak.image_dir, imdb_weak.image_ids{i}),...
        1:Nim, 'UniformOutput', false);
    
    scores = zeros(Nim, prcnn.config.N_parts);
    for i = 1:Nim
        for j=1:prcnn.config.N_parts
            scores(i,j) = detect_boxes{j,prcnn.para.selected_geo_method}(i,end);
        end
    end
    ss = scores(:,1).*scores(:,2).*scores(:,3);
    %[ss,tt]=sort(ss,'descend');
    valid  = ss>.2;
    
    for i = 1 : prcnn.config.N_parts
        weak_fea{i} = extract_deep_feature(prcnn.dir.CNN_MODELS{i}, ...
            prcnn.dir.MODEL_DEF_FILE, imweakpath, detect_boxes{i, prcnn.para.selected_geo_method});
    end
    save(prcnn.dir.WEAK_FEA_FILE, 'weak_fea', 'valid', 'scores', '-v7.3');
end

%% find valid data to add
config = prcnn.config;

% load training features
load(prcnn.dir.TRAIN_FEA_FILE);
TRN_fea = [];
for i = 1 : config.N_parts
    TRN_fea = [TRN_fea train_fea{i}];
end
TRN_fea = scale_feature(TRN_fea);

% load testing feature
load(prcnn.dir.TEST_FEA_FILE);
TST_fea = [];
for i = 1 : config.N_parts
    TST_fea = [TST_fea test_fea{i, prcnn.para.selected_geo_method}];
    %TST_fea = [TST_fea test_fea_gt{i}];
end
TST_fea = scale_feature(TST_fea);

% load weak features
load(prcnn.dir.WEAK_FEA_FILE);
WEAK_fea = [];
for i = 1 : config.N_parts
    WEAK_fea = [WEAK_fea weak_fea{i}];
end
WEAK_fea = scale_feature(WEAK_fea);

% load weak imdb
tmp = load(prcnn.dir.IMDB_WEAK_FILE);
imdb_weak = tmp.imdb;

% load init model
load(prcnn.dir.TRAIN_MODEL_FILE);

%% self paced learning
% !!! we do not use self paced learning in this paper, so Niter is set to 1
K0 = prcnn.para.K0SelfPacedLearning;
eta = prcnn.para.etaSelfPacedLearning;
Niter = prcnn.para.numSelfPacedLearning;
option = prcnn.para.svm_option;

model0 = model;
trainlabel_weak = imdb_weak.labels;
[pd,pe,dc] = predict(trainlabel_weak, sparse(double(WEAK_fea)), model, '-q');
[pd_old,accuracy,~] = predict(config.testlabel, sparse(double(TST_fea)), model, '-q');
acc_test = accuracy(1);
acc_weak = pe(1);
fprintf('Model trained using only strongly-supervised data\n');
fprintf('\tTest accuracy: %.2f%% (%d examples)\n', acc_test, length(config.testlabel));
fprintf('\tWeak accuracy: %.2f%% (%d examples)\n', acc_weak, length(pd));
%{=
for iter = 1:Niter
    K = K0/eta^(iter-1);
    
    [dc_pd,~] = max(dc,[],2);
    dc_gt = arrayfun(@(i)dc(i,trainlabel_weak(i)),1:length(trainlabel_weak));
    dc_gt = dc_gt';
    is_sel = dc_pd-dc_gt<1/K;
    fprintf('Iter %d: K=%.3f, %d positive examples added\n', iter, K, sum(is_sel));
    model_new = train([config.trainlabel; trainlabel_weak(is_sel)]...
        , sparse(double([TRN_fea; WEAK_fea(is_sel,:)])), option);
    model = model_new;
    
    [pd,pe,dc] = predict(trainlabel_weak, sparse(double(WEAK_fea)), model, '-q');
    [pd_new,accuracy,~] = predict(config.testlabel, sparse(double(TST_fea)), model, '-q');
    ltest = config.testlabel;
    %save('predict_results.mat','pd_old','pd_new','ltest');
    acc_test = accuracy(1);
    acc_weak = sum(pd(~is_sel)==trainlabel_weak(~is_sel))/sum(~is_sel)*100;
    fprintf('\tTest accuracy: %.2f%% (%d examples)\n', acc_test, length(config.testlabel));
    fprintf('\tWeak accuracy: %.2f%% (%d examples)\n', acc_weak, sum(~is_sel));
end

disp('Model trained with original CNN features and augmented training data: ');
[pd,pe,dc] = predict(trainlabel_weak, sparse(double(WEAK_fea)), model, '-q');
[~,accuracy,~] = predict(config.testlabel, sparse(double(TST_fea)), model, '-q');
acc_weak = pe(1);
acc_test = accuracy(1);
fprintf('\tTest accuracy: %.2f%% (%d examples)\n', acc_test, length(config.testlabel));
fprintf('\tWeak accuracy: %.2f%% (%d examples)\n', acc_weak, length(trainlabel_weak));
%}



if prcnn.para.isvisualize
    %% show images
    for k=1:200
        is_sel0 = pd==trainlabel_weak;
        [dc_pd,~] = max(dc,[],2);
        dc_gt = arrayfun(@(i)dc(i,trainlabel_weak(i)),1:length(trainlabel_weak));
        dc_gt = dc_gt';
        is_sel1 = dc_pd-dc_gt<1/2;
        is_sel = is_sel0;
        
        clims = find(trainlabel_weak==k);
        added = intersect(find(is_sel), clims);
        added = [added; 0; 0];
        added = [added; intersect(find(is_sel1&~is_sel), clims)];
        fprintf('class %d, %d images added\n', k, length(added));
        Npage = ceil(length(added)/25);
        mkdir(fullfile('weak_added_figure', imdb_weak.classes{k}));
        for j=1:Npage
            bigI = zeros(5*100,5*100,3);
            for m=1:5
                for n=1:5
                    try
                        ii = added((j-1)*25+(m-1)*5+n);
                        im = imread(fullfile(imdb_weak.image_dir, imdb_weak.image_ids{ii}));
                        bigI((m-1)*100+1:m*100, (n-1)*100+1:n*100, :) = imresize(im,[100 100]);
                    catch
                        continue
                    end
                end
            end
            imshow(bigI/256,[]);
            print(gcf, fullfile('weak_added_figure', imdb_weak.classes{k}, sprintf('%d.jpg',j)),'-djpeg');
            %pause;
        end
    end
end
%}

if MODE == 2
    % mode=2: final stage, no need to get more patches for re-training
    return;
end

%{=
%% select valid patches
if ~exist(prcnn.dir.SELECTED_WEAK_PATCH_FILE, 'file')
    [pd,pe,dc] = predict(trainlabel_weak, sparse(double(WEAK_fea)), model0, '-q');
    is_sel = pd==trainlabel_weak;
    score_right = scores(is_sel,:);
    score_wrong = scores(~is_sel,:);
    right_ind = find(is_sel);
    wrong_ind = find(~is_sel);
    
    mright = median(score_right);
    mrong = median(score_wrong);
    
    % final selected samples for re-finetuning cnns
    selected = cell(1,prcnn.config.N_parts);
    for indpart = 1:prcnn.config.N_parts
        wrong2right = find(score_wrong(:,indpart)>mright(indpart)/2);
        right2wrong = find(score_right(:,indpart)<mrong(indpart)/2);
        selected{indpart} = union( setdiff(right_ind, right_ind(right2wrong)), wrong_ind(wrong2right));
    end
    save(prcnn.dir.SELECTED_WEAK_PATCH_FILE, 'selected');
    
    if prcnn.para.isvisualize
        % bbox, head, body
        indpart = 1;
        % for weakly images that is incorrectly classified, ensure that the
        % detecting score exceeds a high value
        % hdr = find(score_wrong(:,indpart)>mright(indpart)/2);
        % length(hdr)

        % for weakly images that is correctly classified, drop examples with the
        % detecting score lower than a low value
        hdr = find(score_right(:,indpart)<mrong(indpart)/2);
        length(hdr)

        load(prcnn.dir.WEAK_DETECT_BOX_FILE);
        R = randperm(length(hdr));
        for jj=1:length(hdr)
            %j = wrong_ind(hdr(R(jj)));
            j = right_ind(hdr(R(jj)));
            im = imread(fullfile(imdb_weak.image_dir, imdb_weak.image_ids{j}));
            clf
            imshow(im, 'Border', 'tight'); hold on
            bb = detect_boxes{indpart,2}(j,:);
            rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
                'EdgeColor', 'g', 'LineWidth', 5);
            pause;
        end
        
        %% show
        for indpart = 1:3
            score_right_selected = find(score_right(:,indpart)>mright(indpart)/2);
            [xx,yy] = sort(score_right(score_right_selected, indpart),'ascend');
            tmp = right_ind(score_right_selected);
            sel_ind = [tmp(yy(end:-1:end-4)); tmp(yy([1 7]))];
            bigI = zeros(100,100*7,3);
            for jj=1:length(sel_ind)
                j = sel_ind(jj);
                im = im2double(imread(fullfile(imdb_weak.image_dir, imdb_weak.image_ids{j})));
                bb = detect_boxes{indpart,2}(j,:);
                bigI(:,(jj-1)*100+1:jj*100,:) = imresize(im(bb(2):bb(4),bb(1):bb(3),:), [100 100]);
            end
            imshow(bigI,[]);
            imwrite(bigI, sprintf('results/figure_select/new/%d.png',indpart));
        end
            
    end
end
end
