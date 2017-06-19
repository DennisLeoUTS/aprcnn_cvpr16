%function create_table_for_paper
% per-cateogy result comparison
clc
load('predict_results.mat');

acc_new = zeros(1,200);
acc_old = zeros(1,200);

% if use mil results
pd_new = pd_mil;

for i=1:200
    ind = ltest==i;
    acc_new(i) = sum(pd_new(ind)==ltest(ind))/sum(ind);
    acc_old(i) = sum(pd_old(ind)==ltest(ind))/sum(ind);
end

Nrow = 40;
Ncol = 5;

fprintf('\\begin{table*}[ht]\n\\begin{center}\n\\footnotesize\n\\begin{tabular}{|')
for i=1:Ncol
    fprintf('l c c|');
end
fprintf('}\n\\hline\n');

for i=1:Ncol
    fprintf(' Name & N & O ');
    if i~=Ncol
        fprintf('&');
    else
        fprintf('\\\\\n\\hline\n');
    end
end

for j=1:200
    cls{j} = strrep(imdb.classes{j},'_','');
end

[~,clsort] = sort(acc_new, 'descend');

for m=1:Nrow
    for n=1:Ncol
        ind = clsort( Nrow*(n-1)+m );
        fprintf('%s & %.0f & %.0f', cls{ind}(1:10), acc_new(ind)*100, acc_old(ind)*100);
        if n~=Ncol
            fprintf(' & ');
        else
            fprintf('\\\\\n');
        end
    end
end

fprintf('\\hline\n\\end{tabular}\n\\end{center}\n\\caption{Per-category classification results.}\n');
fprintf('\\label{tab::comp}\n\\end{table*}\n');

%end