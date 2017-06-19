%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Code to run part-based RCNNs for fine-grained detection %
%%%% Usage: put pretrained deep model paths into cnn_models %%
%%%% Also define model definition file %%%%%%%%%%%%%%%%%%%%%%%
%%%% Change all the paths to your path %%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Zhe Xu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

%% after re-finetuning, using training and weakly-supervised samples together
outFName = 'results/no_finetune.mat';
if ~exist(outFName,'file')
    j=99;  % 99 is for no-finetune cnn net, just for convenience 
    results{j} = run_classification(1, 10000*j);
    find_weak_examples_2add(1, 10000*j);
    save(outFName, 'results');
end


