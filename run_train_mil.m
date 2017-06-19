function accs = run_train_mil(ITER, NUM)
% training function using multi-instance learning on weakly-supervised
% dataset
% written by Zhe Xu in April 2015

%% config
prcnn = config_prcnn(ITER, NUM);

%% load features

% load train feature
load(prcnn.dir.TRAIN_FEA_FILE);

TRN_fea = [];
for i = 1 : prcnn.config.N_parts
    TRN_fea = [TRN_fea train_fea{i}];
end
TRN_fea = scale_feature(TRN_fea);

% imdb for weakly supervised dataset
tmp = load(prcnn.dir.IMDB_WEAK_FILE);
imdb_weak = tmp.imdb;
clear tmp

% load weak features
load(prcnn.dir.WEAK_FEA_MIL_FILE);
WEAK_fea = [];
for i = 1 : prcnn.config.N_parts
    WEAK_fea = [WEAK_fea weak_fea{i}];
end
WEAK_fea = scale_feature(WEAK_fea);
WEAK_fea = arrayfun(@(i)WEAK_fea((i-1)*10+1:i*10,:), 1:size(WEAK_fea,1)/10, 'UniformOutput', false);

% load testing features
load(prcnn.dir.TEST_FEA_FILE);
TST_fea = [];
for i = 1 : prcnn.config.N_parts
    TST_fea = [TST_fea test_fea{i, 2}];
end
TST_fea = scale_feature(TST_fea);

clear weak_fea train_fea


%% train mil
addpath(genpath('MALSAR1.1'));
disp('Train latent SVM ...');


ltrain_s = prcnn.config.trainlabel;
ltest_s  = prcnn.config.testlabel;
ltrain   = imdb_weak.labels;

% strong only
disp('Strongly-supervised data only');
model = train(ltrain_s, sparse(double(TRN_fea)), '-q');
[pd,pe,dc]=predict(ltest_s, sparse(double(TST_fea)), model);
acc1 = pe(1);

fprintf('Accuracy using strong dataset only: %.2f%%\n', acc1)

% select weak samples as the ones which can be correctly classified using
% strong dataset only
wftop = arrayfun(@(i)WEAK_fea{i}(1,:),1:length(WEAK_fea), 'UniformOutput', false);
wftop = cat(1, wftop{:});
[pd,pe,dc]=predict(ltrain, sparse(double(wftop)), model);
ss = find(pd==ltrain);

% strong+top scored weak
WEAK_fea = WEAK_fea(ss);
ltrain   = ltrain(ss);
wftop    = wftop(ss,:);
disp('Strongly-supervised data only + top scored location for weak');
model = train([ltrain_s; ltrain], sparse([double(TRN_fea); wftop]), '-q');
[pd,pe,dc]=predict(ltest_s, sparse(double(TST_fea)), model);
acc2 = pe(1);

fprintf('Accuracy using weak dataset to augment strong dataset: %.2f%%\n', acc2)


% mil
disp('Strongly + weak MIL');
if exist(prcnn.dir.MIL_MODEL_FILE, 'file')
    load(prcnn.dir.MIL_MODEL_FILE);
else
    [w, acc] = mllr_weak_plus_strong(ltrain, WEAK_fea, ltrain_s, TRN_fea, ltest_s, TST_fea);
    save(prcnn.dir.MIL_MODEL_FILE, 'w', 'acc');
end

[~,pd] = max(TST_fea(R3,:)*w,[],2);
acc = sum(pd==ltest_s(R3))/length(pd);
fprintf('Accuracy using multi-instance learning: %.2f%%\n', acc*100)

accs = [acc1, acc2, acc3];


end