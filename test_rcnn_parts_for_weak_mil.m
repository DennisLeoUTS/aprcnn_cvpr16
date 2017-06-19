%% function to run rcnn detections with geometric constaints
%% Three constraints: box, prior and neighbor
%% return boxes which is a cell of N_parts x N_methods
%% Written by Zhe Xu


function boxes = test_rcnn_parts_for_weak_mil(prcnn)

if exist(prcnn.dir.WEAK_DETECT_BOX_MIL_FILE, 'file') && prcnn.para.canSkip
    load(prcnn.dir.WEAK_DETECT_BOX_MIL_FILE);
    boxes = detect_boxes;
end

% number of top mil windows
Ntop = 10;

% load imdb
load(prcnn.dir.IMDB_WEAK_FILE);

figure_dir = prcnn.dir.figure_dir;

load(prcnn.dir.GEO_PRIOR_FILE);
feat_opts.prior = prior;
feat_opts.X = X;
config = prcnn.config;

boxes = cell(config.N_parts, config.N_methods);
for i = 1 : config.N_parts
    for j  = 1 : config.N_methods
        boxes{i,j} = -1 * ones(length(config.impathtest), Ntop, 5);
    end
end

% TODO are these arbitrary numbers legit?
top_scores = cell(1, config.N_methods);
thresh = -inf(1, config.N_methods);
box_counts = zeros(1, config.N_methods);

% load rcnn detector results for testing images
parts = config.parts;
% for i=1:3
%     s{i} = load(sprintf('rcnn/feat_cache/v1_finetune_cub_train_%s_iter_10k/parts_test/rcnn_scores_1to5794.mat',parts{i}));
% end

nim = length(imdb.image_ids);

% TODO fix num_batches
for i = 1 : nim
    pdFile = sprintf(prcnn.dir.WEAK_DETECT_BOX_IMAGE_MIL_FILE, parts{1}, imdb.name, imdb.image_ids{i});
    if exist(pdFile, 'file') && prcnn.para.canSkip
        load(pdFile);
        for m = 1 : config.N_methods
            for k = 1 : config.N_parts
                boxes{k,m}(i, :, :) = pdbox{k,m};
            end
        end
        continue
    end
    
    fprintf('test %d/%d\n', i, nim);
    %try
    d = load(sprintf(prcnn.dir.RCNN_DETECT_SCORE_FILE,...
        parts{1}, imdb.name, imdb.image_ids{i}));
    zs = d.score;
    boxes_ = d.boxes;
    
    zp = [];
    for j=1:length(parts)-1
        d = load(sprintf(prcnn.dir.RCNN_DETECT_SCORE_FILE,...
            parts{j+1}, imdb.name, imdb.image_ids{i}));
        zp = [zp d.score];
    end
    
    
    % find knn_idx on the fly
    %[~, best_root_guess] = max(zs);
    %knn_idx = knnsearch(feat_opts.train_fea, d.feat(best_root_guess,:), 'K', 30);
    
    zs_ = zs;
    zs_n = exp(zs_) ./ (1 + exp(zs_));
    
    zp_n = zeros(length(zs), length(parts) - 1);
    for p = 1 : length(parts) - 1
        zp_n(:,p) = exp(zp(:,p)) ./ (1 + exp(zp(:,p)));
    end
    
    scores = ones(length(zs_n), config.N_parts+1, config.N_methods);
    scores_idx = zeros(length(zs_), length(parts) - 1, config.N_methods);
    max_ps = zeros(length(zs_), length(parts) - 1);
    %neighbor_prior = fit_neighbors(knn_idx, feat_opts.X);
    
    for k = 1 : length(zs_)
        % fix one root filter
        w = double(boxes_(k,3) - boxes_(k,1));
        h = double(boxes_(k,4) - boxes_(k,2));
        
        % box constraint
        I_p = find(boxes_(:,1) >= boxes_(k,1) - prcnn.para.geo_box_gap & boxes_(:,2) >= boxes_(k,2)-prcnn.para.geo_box_gap ...
            & boxes_(:,3) <= boxes_(k,3) + prcnn.para.geo_box_gap & boxes_(:,4) <= boxes_(k,4) + prcnn.para.geo_box_gap);
        zpn_ = zp_n(I_p, :);
        s_idx = scores_idx(k, :, :);
        s_ = scores(k, :, :);
        s_(1,1,1) = zs_n(k);
        s_(1,2,1) = zs_n(k);
        for p = 1 : length(parts) - 1
            [max_s, argmax_p] = max(zpn_(:,p));
            s_(1,p+2,1)= max_s;
            s_(1,1,1) = s_(1,1,1) * max_s;
            s_idx(1, p, 1) = I_p(argmax_p);
        end
        
        % normalize boxes and change the format to [center_x center_y width height]
        n_boxes = double(boxes_(I_p, :)- repmat([boxes_(k, 1) boxes_(k, 2) boxes_(k, 1) boxes_(k, 2)], length(I_p), 1)) ...
            ./ repmat([w h w h], length(I_p), 1);
        n_boxes = [(n_boxes(:, 1) + n_boxes(:, 3)) / 2 (n_boxes(:, 2) + n_boxes(:, 4)) / 2 ...
            n_boxes(:, 3) - n_boxes(:, 1)  n_boxes(:, 4) - n_boxes(:, 2)];
        
        % prior constraint
        s_(1,1,2) = zs_n(k);
        s_(1,2,2) = zs_n(k);
        for p = 1 : length(parts) - 1
            zz = zp_n(I_p, p) .* (pdf(feat_opts.prior{p}, n_boxes) .^ prcnn.para.geo_mg_alpha);
            [max_p, argmax_p] = max(zz);
            s_(1,p+2,2)= max_p;
            s_(1,1,2) = s_(1,1,2) * max_p;
            max_ps(k,p) = max_p;
            s_idx(1, p, 2) = I_p(argmax_p);
        end
        
        % neighbor constraint
        %         for p = 1 : length(part_models) - 1
        %             zz = zp_n(I_p, p) .* (pdf(neighbor_prior{p}, n_boxes) .^ 0.01);
        %             [max_p, argmax_p] = max(zz);
        %             s_(3) = s_(3) * max_p;
        %             s_idx(1,p,3) = I_p(argmax_p);
        %         end
        
        scores_idx(k,:,:) = s_idx;
        scores(k,:,:) = s_;
    end
    for m = 1 : config.N_methods
        [max_scores, argmaxs] = sort(scores(:, 1, m), 'descend');
        I = find(scores(:, 1, m) > thresh(m));
        [~, ord] = sort(scores(I, 1, m), 'descend');
        box_counts(m) = box_counts(m) + length(ord);
        top_scores{m} = cat(1, top_scores{m}, scores(I(ord),1,m));
        top_scores{m} = sort(top_scores{m}, 'descend');
        
        for jj=1:min(Ntop,length(argmaxs))
            argmax = argmaxs(jj);
            boxes{1,m}(i, jj, :) = [boxes_(argmax, :) scores(argmax,2,m)];
            for p = 1 : length(parts) - 1
                boxes{p+1,m}(i, jj, :) = [boxes_(scores_idx(argmax, p, m), :) scores(argmax,p+2,m)];
            end
        end
    end
    
    % filter out detections below thresh and fill with -1
%     for m = 1 : config.N_methods
%         for k = 1 : config.N_parts
%             I = find(boxes{k,m}(:, end) < thresh(m));
%             if ~isempty(I)
%                 fprintf('below thresh: %d/%d\n', length(I), i);
%                 %xx = length(I);
%                 %boxes{k,m}(I,:) = repmat([-1 -1 -1 -1 -1], xx, 1);
%             end
%             %boxes{k}{m} = boxes{k}{m}(:,1:4);
%         end
%     end
    
    % save results
    pdbox = cell(config.N_parts, config.N_methods);
    for m = 1 : config.N_methods
        for k = 1 : config.N_parts
            pdbox{k,m} = boxes{k,m}(i, :, :);
        end
    end
    save(pdFile, 'pdbox');
    
    %visualize results
    if prcnn.para.isvisualize
        %dif = sum(boxes{1,2}(i,:)-boxes{1,1}(i,:)) + ...
        %sum(boxes{2,2}(i,:)-boxes{2,1}(i,:))+sum(boxes{3,2}(i,:)-boxes{3,1}(i,:));
        %if (abs(dif)<1)
        %    continue
        %end

        clf
        subplot(1,2,1),
        im = imread(fullfile(imdb.image_dir, imdb.image_ids{i}));
        imshow(im, 'Border', 'tight');
        hold on,
        bb = boxes{1,2}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'r', 'LineWidth', 5);
        text(bb(1),bb(2)+5,sprintf('bbox: %.4f', bb(5)),...
            'BackgroundColor', 'r', 'Color', 'w', 'FontSize', 15);
        bb = boxes{2,2}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'g', 'LineWidth', 2);
        text(bb(1),bb(2)+5, 'head',...
            'BackgroundColor', 'g', 'FontSize', 15);
        bb = boxes{3,2}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'b', 'LineWidth', 2);
        text(bb(1),bb(2)+5, 'body',...
            'BackgroundColor', 'b', 'Color', 'w', 'FontSize', 15);

        subplot(1,2,2),
        im = imread(fullfile(imdb.image_dir, imdb.image_ids{i}));
        imshow(im, 'Border', 'tight');
        hold on,
        bb = boxes{1,1}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'r', 'LineWidth', 5);
        text(bb(1),bb(2)+5,sprintf('bbox: %.4f', bb(5)),...
            'BackgroundColor', 'r', 'Color', 'w', 'FontSize', 15);
        bb = boxes{2,1}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'g', 'LineWidth', 2);
        text(bb(1),bb(2)+5, 'head',...
            'BackgroundColor', 'g', 'FontSize', 15);
        bb = boxes{3,1}(i,:);
        rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
            'EdgeColor', 'b', 'LineWidth', 2);
        text(bb(1),bb(2)+5, 'body',...
            'BackgroundColor', 'b', 'Color', 'w', 'FontSize', 15);
        %pause;

        outFName = fullfile(figure_dir, imdb.image_ids{i});
        [tmp1,tmp2,~]=fileparts(outFName);
        mkdir_if_missing(fullfile(tmp1));
        outFName = fullfile(tmp1,[tmp2,'.png']);
        print(gcf, '-dpng', outFName);
    end
end

detect_boxes = boxes;
save(prcnn.dir.WEAK_DETECT_BOX_MIL_FILE, 'detect_boxes');
end
