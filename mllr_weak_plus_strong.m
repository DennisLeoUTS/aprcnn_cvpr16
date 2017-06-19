function [w, acc] = mllr_weak_plus_strong(ltrain, ftrain, ...
    ltrain_s, ftrain_s, ltest_s, ftest_s, param_range, rho_L2_range, active_pos_ind)
% multiclass multitask mllr using malsar

% input:
% X: {nlatent * D} * N
% Y: 1 * N
% number of tasks: Nclass
% W: Nclass * D
%{=

if nargin<9
    param_range = 0;
end

if nargin<10
    rho_L2_range = 1e-3;
end

if nargin<11
    active_pos_ind = ones(size(ltrain));
end

%param_range = [1e-3 1e-2 1e-1];
%param_range = 1e-3;
%rho_L2_range = [1e1 1e0 1e-1 1e-2 1e-3];
%rho_L2_range = [1e-3];
%l2 = 1;

[~,~,ltrain] = unique(ltrain);
[~,~,ltrain_s] = unique(ltrain_s);
[~,~,ltest_s] = unique(ltest_s);
ltrain_s = ltrain_s';
ltest_s = ltest_s';
ltrain = ltrain';

for l1 = 1:length(param_range)
    for l2 = 1:length(rho_L2_range)
        pos_latent = active_pos_ind;

        %param = 0;
        param = param_range(l1);
        rho_L2 = rho_L2_range(l2);

        w = mllr_malsar_ws(ltrain, ftrain, pos_latent, ltrain_s, ftrain_s, ltest_s, ftest_s, param, rho_L2 );

        %{=
        %[test_score, active_hidden] = cellfun(@(x)max(x*w,[],1), ftest, 'UniformOutput', false);
        ts = ftest_s*w;
        [tss, pd]=max(ts, [], 2);
        acc = sum(pd==ltest_s')/length(pd);
        %}
    end

end

end
