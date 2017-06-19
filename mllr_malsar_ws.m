function W = mllr_malsar_ws(ltrain, ftrain, pos_latent, ltrains, ftrains, ltests, ftests, param, rho_L2 )

% optimization options
opts = [];
opts.maxIter = 50;

tic;

% build model using the optimal parameter 
opts.rand_hidden = 1;
opts.init=2;
nclass = length(unique(ltrain));
%nclass = 200;
init_hidden = ones(length(ltrain), nclass);
for i=1:length(ltrain)
    nlatent = size(ftrain{i},1);
    init_hidden(i,:) = ceil(nlatent*rand(1,nclass));
    for j=1:length(unique(ltrain))
        if ltrain(i)==j
            init_hidden(i,j)=pos_latent(i);
        end
    end
end
opts.init_hidden = init_hidden;
opts.rho_L2 = rho_L2;
%param = 0;

for outer_loop = 1:5
%W = MLLR_Trace(X, Y, param, opts);
    [W, funcVal] = MLLR_L21_ws(ftrain, ltrain, ftrains, ltrains, param, opts);
    
    ts = ftests*W;
    [tss, pd]=max(ts, [], 2);
    acc = sum(pd==ltests')/length(pd);
    
    % relabel positive examples
    [tt, ti] = cellfun(@(x)max(x*W,[],1), ftrain, 'UniformOutput', false);
    init_hidden_new = cat(1,ti{:});
    Nchanged = 0;
    for i=1:length(ltrain)
        if init_hidden(i,ltrain(i))~=init_hidden_new(i,ltrain(i));
            Nchanged = Nchanged+1;
        end
        init_hidden(i,ltrain(i))=init_hidden_new(i,ltrain(i));
    end
    fprintf('OuterLoop %d: %d pos latent changes, funcVal: %.4f, testing accuracy: %.2f%%\n', ...
        outer_loop, Nchanged, funcVal(end), acc*100);
    opts.init = 1;
    opts.W0 = W;
end
%W = MLLR_L21(X, Y, param, opts);

%{
[test_score, ~] = cellfun(@(x)max(x*W,[],1), ftest, 'UniformOutput', false);
ts = cat(1,test_score{:});
[tss, pd]=max(ts, [], 2);
acc = sum(pd==ltest')/length(pd);
%}
toc;

end
