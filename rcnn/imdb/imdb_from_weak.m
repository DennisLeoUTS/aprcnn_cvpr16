function imdb = imdb_from_weak()
% get imdb for weakly supervised dataset downloaded from the Internet

conf = rcnn_config;

cache_file = conf.dir.IMDB_WEAK_FILE;
if exist(cache_file,'file')
    load(cache_file);
    return;
end
%% get images
weakDir = conf.dir.WEAR_DIR;
weakFileNames = cell(0);
weakLabels = [];
classes = cell(0);

NperClass = 100;  % 100 images per class

cls = dir(weakDir);
for j=3:length(cls)
    cl = cls(j).name;
    classes = cat(1,classes,cl);
    clDir = fullfile(weakDir, cl);
    ims = dir(clDir);
    ims = ims(3:end);
    S = [ims(:).datenum].'; % you may want to eliminate . and .. first.
    [~,S] = sort(S, 'descend');   % find latest uploaded files (earliest downloaded)
    ims = {ims(S).name};
    ims = arrayfun(@(i)fullfile(cl,ims{i}),1:length(ims),'UniformOutput',false);
    nim = min(NperClass, length(ims));
    weakFileNames = cat(1, weakFileNames, ims(1:nim)');
    weakLabels = cat(1, weakLabels, (j-2)*ones(nim,1));
end

%% get imdb
imdb.name = 'weak';
imdb.image_dir = weakDir;
imdb.image_ids = weakFileNames;
imdb.classes = classes;
imdb.num_classes = length(imdb.classes);
imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
imdb.class_ids = 1:imdb.num_classes;
imdb.image_at = @(i)sprintf('%s/%s', imdb.image_dir, imdb.image_ids{i});
imdb.labels = weakLabels;
for i = 1:length(imdb.image_ids)
    info = imfinfo(sprintf('%s/%s', imdb.image_dir, imdb.image_ids{i}));
    imdb.sizes(i, :) = [info.Height info.Width];
end


fprintf('Saving imdb to cache...');
save(cache_file, 'imdb', '-v7.3');
fprintf('done\n');
end
