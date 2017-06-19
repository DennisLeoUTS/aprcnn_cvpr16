%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Code to run part-based RCNNs for fine-grained detection %
%%%% Usage: put pretrained deep model paths into cnn_models %%
%%%% Also define model definition file %%%%%%%%%%%%%%%%%%%%%%%
%%%% Change all the paths to your path %%%%%%%%%%%%%%%%%%%%%%%
%%%% Written by Zhe Xu %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

%% before re-finetuning, using training samples only
results_part_rcnn = run_classification(1, 10000);

find_weak_examples_2add(1, 10000);

%% after re-finetuning, using training and weakly-supervised samples together
if prcnn.para.isuse_mil
    results = run_classification_mil(2, 20000);
    
    run_train_mil(2, 20000);
else
    results = run_classification(2, 20000);

    find_weak_examples_2add(2, 20000);
end

