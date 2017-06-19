function fea = scale_feature(fea)
ppp = 0.3;
for i = 1:size(fea,2)
    fea(:,i) = sign(fea(:,i)).*abs(fea(:,i)).^ppp;
end
end