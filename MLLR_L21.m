%% FUNCTION Logistic_TGL
%  L21 Joint Feature Learning with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + opts.rho_L2 * \|W\|_2^2 + rho1 * \|W\|_{2,1} }
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: L2,1-norm group Lasso parameter.
%   OPTIONAL
%      opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
%   W: model: d * t
%   C: model: 1 * t
%   funcVal: function value vector.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jun Liu and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% RELATED PAPERS
%
%   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
%   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
%       Report, 2010.
%
%% RELATED FUNCTIONS
%  Least_L21, init_opts

%% Code starts here
function [W, funcVal] = MLLR_L21(X, Y, rho1, opts)

if nargin <3
    error('\n Inputs: X, Y, rho1, should be specified!\n');
end
%X = multi_transpose(X);

if nargin <4
    opts = [];
end



% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

if opts.rand_hidden==2
    nclass = length(unique(Y));
    for ii=1:length(X)
        nlatent = size(X{ii},1);
        active_hidden(ii,:) = ceil(nlatent*rand(1,nclass));
    end
elseif opts.rand_hidden==0
    nclass = length(unique(Y));
    l = length(X);
    active_hidden = ones(l, nclass);
else
    active_hidden = opts.init_hidden;
end

[~,~,Y] = unique(Y);
task_num  = length (unique(Y));
dimension = size(X{1}, 2);
funcVal = [];


if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init== 0
    W0 = randn(dimension, task_num);
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0 = zeros(dimension, task_num);
    end
end



bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;
iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    [gWs, Fs ]  = gradVal_eval(Ws);
    
    % the Armijo Goldstein line search scheme
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        r_sum = (nrm_delta_Wzp)/2;
        
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + trace(delta_Czp' * gCs)...
        %             + gamma/2 * nrm_delta_Wzp ...
        %             + gamma/2 * nrm_delta_Czp;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp.* gWs))...
            + gamma/2 * nrm_delta_Wzp;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1));
    %if mod(iter,1)==0, fprintf('Iter %d: funcVal: %.4f\n', iter, funcVal(end)); end
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [Wp] = FGLasso_projection (W, lambda )
        % solve it in row wise (L_{2,1} is row coupled).
        % for each row we need to solve the proximal opterator
        % argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        
        Wp = zeros(size(W));
        if opts.pFlag
            %             % 1. SEGMENT by feature dimensionality
            %             s_num = opts.pSeg_num;
            %             seg_len = ceil(dimension / s_num);
            %             idx_seg = cell(s_num, 1);
            %             W_seg   = cell(s_num, 1);
            %             for s_idx = 1: s_num
            %                 idx_seg{s_idx} = gen_seg_idx(s_idx, seg_len, dimension);
            %                 W_seg{s_idx} = W(idx_seg{s_idx}, :);
            %             end
            %             Wp_seg  = cell(s_num, 1);
            %             % 2. MAP
            %             parfor s_idx = 1 : s_num
            %                 Wp_seg{s_idx} = zeros(length(idx_seg{s_idx}), size(W, 2));
            %
            %                 for i = 1: length(idx_seg{s_idx})
            %                     v = W_seg{s_idx}(i, :);
            %                     nm = norm(v, 2);
            %                     if nm == 0
            %                         w = zeros(size(v));
            %                     else
            %                         w = max(nm - lambda, 0)/nm * v;
            %                     end
            %                     Wp_seg{s_idx}(i, :) = w';
            %                 end
            %             end
            %             % 3. REDUCE
            %             for s_idx = 1: s_num
            %                 Wp(idx_seg{s_idx}, :) = Wp_seg{s_idx};
            %             end
            parfor i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        else
            for i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        end
    end


% private functions
    function [grad_W, funcVal] = gradVal_eval(W)
        grad_W = zeros(dimension, task_num);
        l = length(X);
        active_score = zeros(size(active_hidden));
        for i=1:l
            xx = X{i}*W;
            [val, max_id] = max(xx,[],1);
            for j=1:task_num
                if Y(i)~=j
                    active_score(i,j) = val(j);
                    active_hidden(i,j) = max_id(j);
                else
                    active_score(i,j) = xx(active_hidden(i,j), j);
                end
            end
        end
        exp_wTx = exp(active_score);
        
        exp_wTx_pos = zeros(l,1);
        for i=1:l
            exp_wTx_pos(i)=exp_wTx(i,Y(i));
        end
        %exp_wTx_pos = arrayfun(@(i)exp_wTx(i,y(i)), 1:l, 'UniformOutput', 'false');
        pall = exp_wTx./repmat(sum(exp_wTx,2), 1, task_num);
        pyi = exp_wTx_pos./sum(exp_wTx,2);
        
        for i=1:l
            for tt = 1:task_num
                p = -pall(i,tt);
                if Y(i)==tt
                    p = 1+p;
                end
                grad_W(:,tt) = grad_W(:,tt) - X{i}(active_hidden(i,tt), :)'*p/l;
            end
        end
        grad_W = grad_W+ rho_L2 * 2 * W;
        % here when computing function value we do not include
        % l1 norm.
        funcVal = -sum(log(pyi))/l + rho_L2 * norm(W, 'fro')^2;
    end

    function [funcVal] = funVal_eval (W)
        l = length(X);
        active_score = zeros(size(active_hidden));
        for i=1:l
            xx = X{i}*W;
            [val, max_id] = max(xx,[],1);
            for j=1:task_num
                if Y(i)~=j
                    active_score(i,j) = val(j);
                    active_hidden(i,j) = max_id(j);
                else
                    active_score(i,j) = xx(active_hidden(i,j), j);
                end
            end
        end
        
        active_score = active_score - repmat(max(active_score,[],2), 1, size(active_score,2));
        exp_wTx = exp(active_score);
        exp_wTx_pos = zeros(l,1);
        for i=1:l
            exp_wTx_pos(i)=exp_wTx(i,Y(i));
        end
        pyi = exp_wTx_pos./sum(exp_wTx,2);
        % here when computing function value we do not include
        % l1 norm.
        funcVal = -sum(log(pyi))/l + rho_L2 * norm(W, 'fro')^2;
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(w, 2);
        end
    end
end

function seg_ind_set = gen_seg_idx(s_idx, seg_len, total_len)
%generate segmentation indices.
seg_ind_set = (s_idx-1) * seg_len + 1  : min(s_idx* seg_len, total_len);
end

function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y)
%gradient and logistic evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_c = sum(b);
grad_w = x * b;
end

function [ funcVal ] = unit_funcVal_eval( w, c, x, y)
%function value evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end