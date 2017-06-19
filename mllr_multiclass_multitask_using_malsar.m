function [w, acc] = mllr_multiclass_multitask_using_malsar(ltrain, ftrain, ltest, ftest,...
    param_range, rho_L2_range, active_pos_ind)
% multiclass multitask mllr using malsar

% input:
% X: {nlatent * D} * N
% Y: 1 * N
% number of tasks: Nclass
% W: Nclass * D
%{=

%param_range = [1e-3 1e-2 1e-1];
%param_range = 1e-3;
%rho_L2_range = [1e1 1e0 1e-1 1e-2 1e-3];
%rho_L2_range = [1e-3];
%l2 = 1;

[~,~,ltrain] = unique(ltrain);
[~,~,ltest] = unique(ltest);
ltrain = ltrain';
ltest = ltest';
nclass = max(ltrain);

for l1 = 1:length(param_range)
    for l2 = 1:length(rho_L2_range)
        pos_latent = active_pos_ind;
        %pos_latent = ones(size(active_pos_ind));

        %param = 0;
        param = param_range(l1);
        rho_L2 = rho_L2_range(l2);

        %W = mllr_malsar_with_prior(ltrain, ftrain, ltest, ftest, pos_latent, prior_train, prior_test, param, rho_L2 );
        w = mllr_malsar(ltrain, ftrain, ltest, ftest, pos_latent, param, rho_L2 );

        %{=
        [test_score, active_hidden] = cellfun(@(x)max(x*w,[],1), ftest, 'UniformOutput', false);

        ts = cat(1,test_score{:});
        [tss, pd]=max(ts, [], 2);
        acc = sum(pd==ltest')/length(pd);
        %}
    end

end

end