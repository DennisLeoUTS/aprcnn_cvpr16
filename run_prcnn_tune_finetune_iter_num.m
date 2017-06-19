%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Code to run part-based RCNNs for fine-grained detection %
%%%% Usage: put pretrained deep model paths into cnn_models %%
%%%% Also define model definition file %%%%%%%%%%%%%%%%%%%%%%%
%%%% Change all the paths to your path %%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Zhe Xu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

%% after re-finetuning, using training and weakly-supervised samples together
outFName = 'results/tune_iter_finetune.mat';
if ~exist(outFName,'file')
    for j=1:8
        results{j} = run_classification(2, 10000*j);
        acc_test = find_weak_examples_2add(2, 10000*j);
        results{j}.accuracy_after_weak = acc_test;
        save(outFName, 'results');
    end
    save(outFName, 'results');
else
    load(outFName);
end

re=arrayfun(@(i)results{i}.detected_accuracy(1),1:8)
plot(re)
re2=arrayfun(@(i)results{i}.accuracy_after_weak,1:8)
hold on
plot(re2,'r');

