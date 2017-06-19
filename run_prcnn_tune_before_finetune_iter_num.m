%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Code to run part-based RCNNs for fine-grained detection %
%%%% Usage: put pretrained deep model paths into cnn_models %%
%%%% Also define model definition file %%%%%%%%%%%%%%%%%%%%%%%
%%%% Change all the paths to your path %%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Zhe Xu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

%% after re-finetuning, using training and weakly-supervised samples together
outFName = 'results/tune_iter_finetune_before.mat';
results = {};
if ~exist(outFName,'file')
    for j=1:6
        results{j} = run_classification(1, 10000*j);
        %acc_test = find_weak_examples_2add(2, 10000*j);
        %results{j}.accuracy_after_weak = acc_test;
        save(outFName, 'results');
    end
    save(outFName, 'results');
else
    load(outFName);
end


re=arrayfun(@(i)results{i}.detected_accuracy(1),1:6);
re = [67.28 re];
plot(re, 'o-', 'LineWidth', 3, 'MarkerSize', 10)
set(gca,'FontSize',20)
xlabel('# of Iterations (x10k)')
ylabel('Accuracy')


%re2=arrayfun(@(i)results{i}.accuracy_after_weak,1:6)
%hold on
%plot(re2,'r');

