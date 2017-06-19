%function provide_figures_for_paper

%% GET IMDB
% imdb_train = imdb_from_voc('train');
% trainIms = imdb_train.image_ids;
% P = randperm(length(trainIms));
% trainRoidb = imdb_train.roidb_func(imdb_train);

%{
%% GET FULL TRAINING IMAGES 
for j=1:10
    outFName = sprintf('figures/%d.png',P(j));
    close all
    im = imread(imdb_train.image_at(P(j)));
    sz = size(im);
    if sz(1)>sz(2)
        continue
    end
    imshow(im, 'Border', 'tight');
    hold on,
    bb = trainRoidb.rois(P(j)).boxes;
    cc = trainRoidb.rois(P(j)).class;
    br = bb(cc==1,:);
    rectangle('Position', [br(1),br(2),br(3)-br(1),br(4)-br(2)],...
        'EdgeColor', 'r', 'LineWidth', 5);
    bd = bb(cc==2,:);
    rectangle('Position', [bd(1),bd(2),bd(3)-bd(1),bd(4)-bd(2)],...
        'EdgeColor', 'y', 'LineWidth', 3);
    bh = bb(cc==3,:);
    rectangle('Position', [bh(1),bh(2),bh(3)-bh(1),bh(4)-bh(2)],...
        'EdgeColor', 'y', 'LineWidth', 3);
    print(gcf, '-dpng', outFName);
end
%}

%{
%% GET PART PATCHES
%Nrows = [5,2,2];
%Ncolumns = [2,5,5];
%w = 480;
%h = 360;
Nrows = [3,1,1];
Ncolumns = [1,3,3];
w = 120;
h = 90;

for p=1:3
    outFName = sprintf('figures/part_%d_2.png',p);
    close all
    Nrow = Nrows(p);
    Ncolumn = Ncolumns(p);
    P = randperm(length(trainIms));
    bigI = zeros(Nrow*h, Ncolumn*w, 3);
    for i=1:Ncolumn
        for j=1:Nrow
            t = (i-1)*Nrow+j;
            im = imread(imdb_train.image_at(P(t)));
            im = im2double(im);
            bb = trainRoidb.rois(P(t)).boxes;
            cc = trainRoidb.rois(P(t)).class;
            bb = bb(cc==p,:);
            ims = im(bb(2)+1:bb(4)+1, bb(1)+1:bb(3)+1, :);
            ims = imresize(ims, [h w]);
            bigI((j-1)*h+1:j*h, (i-1)*w+1:i*w, :) = ims;
        end
        % imshow(ims,[]);
    end
    %imshow(bigI);
    %print(gcf, '-depsc', outFName);
    imwrite(bigI, outFName);
end
%}

%% get weak imdb
%imdb_weak = imdb_from_voc('weak');
%Nim_weak = length(imdb_weak.image_ids);

%{
%% WEAK FULL IMAGES
Nrow = 6;
Ncolumn = 3;
w = 120;
h = 90;
outFName = 'figures/weak_1.png';
bigI = zeros(Nrow*h, Ncolumn*w, 3);
P = randperm(Nim_weak);
for i=1:Ncolumn
    for j=1:Nrow
        t = (i-1)*Nrow+j;
        im = imread(imdb_weak.image_at(P(t)));
        im = im2double(im);
        ims = imresize(im, [h w]);
        bigI((j-1)*h+1:j*h, (i-1)*w+1:i*w, :) = ims;
        imwrite(im,sprintf('figures/weak/%d.jpg', t));
    end
end
%imshow(bigI,[]);
imwrite(bigI, outFName);
%}

%{=
%% WEAK IMAGES PARTS PATCHES
%Nrows = [5 4 4];
%Ncolumns = [4 5 5];
%Nrows = [3 2 2];
%Ncolumns = [2 3 3];
Nrows = [10,10,10];
Ncolumns = [5,5,5];
w = 120;
h = 90;
load('../caches/rcnn_detect_boxes_weak.mat');
load('../caches/selected_weak_samples.mat');
tmp_ind = Nim_weak;

for p=1:3
    outFName = sprintf('figures/weak_part_%d_big.png',p);
    close all
    P = randperm(Nim_weak);
    Nrow = Nrows(p);
    Ncolumn = Ncolumns(p);
    bigI = zeros(Nrow*h, Ncolumn*w, 3);
    for i=1:Ncolumn
        for j=1:Nrow
            t = (i-1)*Nrow+j;
            while isempty(find(selected{p}==P(t), 1))
                tmp = P(tmp_ind);
                P(tmp_ind) = P(t);
                P(t) = tmp;
                tmp_ind = tmp_ind-1;
            end
            im = imread(imdb_weak.image_at(P(t)));
            im = im2double(im);
            bb = detect_boxes{p,1}(P(t),:);
            ims = im(bb(2):bb(4), bb(1):bb(3), :);
            ims = imresize(ims, [h w]);
            bigI((j-1)*h+1:j*h, (i-1)*w+1:i*w, :) = ims;
        end
        % imshow(ims,[]);
    end
    %imshow(bigI);
    imwrite(bigI, outFName);
end
%}


%% noise removal

%end