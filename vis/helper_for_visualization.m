function helper_for_visualization(i, prcnn, pd, imdb_test, imdb_train, model, ...
    TST_fea, TRN_fea, trainValid, detect_boxes, rois_train, MODE, ...
    imdb_weak, detect_boxes_after, WEAK_fea_after)
% helper function for create_visualization_for_paper.m

if nargin>=13
    refine = 1;
else
    refine = 0;
end

colors = {'r','y','g'};
fprintf('GT: %d, PD: %d, ', prcnn.config.testlabel(i), pd(i));
im = imread(imdb_test.image_at(i));

h{1}=figure(1);
clf, imshow(im, 'Border', 'Tight'), hold on
for k=1:3
    bb = detect_boxes{k,2}(i,:);
    rectangle('Position', [bb(1),bb(2),bb(3)-bb(1),bb(4)-bb(2)],...
       'EdgeColor', colors{k}, 'LineWidth', 5);
    
%     i1=im(bb(2):bb(4),bb(1):bb(3),:);
%     i1=imresize(i1,[200 200]);
%     clf,imshow(i1,'Border','Tight');
%     print(gcf, '-djpeg', sprintf('results/figure_nn/%04d/p%d.jpg', i, k));
end

imFeat = TST_fea(i,:);
dc = imFeat*model.w';
[~,pds] = sort(dc,'descend');
fprintf('gt position: #%d\n', find(pds == prcnn.config.testlabel(i)));

for k=1:3
    imFeatPart = TST_fea(i,(k-1)*4096+1:k*4096);
    trDist = pdist2(imFeatPart, TRN_fea(trainValid{k},(k-1)*4096+1:k*4096));
    Ntrain = length(trDist);
    
    if refine
        trDist_after = pdist2(imFeatPart, WEAK_fea_after(:,(k-1)*4096+1:k*4096));
        tmp = [trDist trDist_after];
        [di, indtr] = sort(tmp, 'ascend');
        for j=1:10
            if indtr(j)<=Ntrain
                indtr(j) = trainValid{k}(indtr(j));
            end
        end
    else
        [di, indtr] = sort(trDist, 'ascend');
        indtr = trainValid{k}(indtr);
    end
    
    bigI = zeros(300,300,3);
    for m=1:3
        for n=1:3
            t = m*3+n-3;
            
            if indtr(t)>Ntrain && refine
                im1 = imdb_weak.image_at(indtr(t)-Ntrain);
                bb_label = imdb_weak.labels(indtr(t)-Ntrain);
            else
                im1 = imdb_train.image_at(indtr(t));
                
                bb_label = prcnn.config.trainlabel(indtr(t));
            end
            fprintf('%d, ', bb_label);
            
            if refine && indtr(t)>length(trDist) 
                bb = detect_boxes_after{k,2}(indtr(t)-Ntrain,:);
            else
                roi = rois_train(indtr(t));
                bb = roi.boxes(roi.class==k,:);
                bb = bb+1;
            end
            
            im1 = im2double(imread(im1));
            if size(im1,3)==1
                im1 = gray2rgb(im1);
            end
            
            bb=max(bb,1);
            bb(4)=min(bb(4),size(im1,1));
            bb(3)=min(bb(3),size(im1,2));
            im1 = imresize(im1(bb(2):bb(4),bb(1):bb(3),:), [100 100]);
            
            if bb_label==prcnn.config.testlabel(i)
                if indtr(t)>Ntrain && refine
                    co = [1 1 0];
                else
                    co = [0 1 0];
                end
                im1([1:5,96:100],:,:) = repmat(reshape(co,1,1,3), [10 100 1]);
                im1(:,[1:5,96:100],:) = repmat(reshape(co,1,1,3), [100 10 1]);
            elseif bb_label==pd(i)
                im1([1:5,96:100],:,:) = repmat(reshape([1 0 0],1,1,3), [10 100 1]);
                im1(:,[1:5,96:100],:) = repmat(reshape([1 0 0],1,1,3), [100 10 1]);
            end
                

            bigI((m-1)*100+1:m*100,(n-1)*100+1:n*100,:) = im1;
        end
    end
    fprintf('\n');
    h{k+1}=figure(k+1);clf,
    imshow(bigI, 'Border', 'Tight');
end

for k=1:4
    if MODE>=2 && k==1
        continue
    end
    outFName = sprintf('results/figure_nn/%04d/%d_%d_%d.jpg', i, i, MODE, k);
    [yy,~,~]=fileparts(outFName);
    if ~exist(yy,'dir')
        mkdir(yy);
    end
    print(h{k}, '-djpeg', outFName);
    
    outFName = sprintf('results/figure_nn/%04d/%d_%d_%d.eps', i, i, MODE, k);
    print(h{k}, '-depsc', outFName);
end