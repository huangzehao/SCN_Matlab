function fea = extrconvfea(im, filters)
[H,W] = size(im);
nf = size(filters,2);
fs = round(sqrt(size(filters,1)));
hfs = floor(fs / 2);
fea = zeros(H-fs+1,W-fs+1,nf);
for i = 1:nf
    filter = reshape(filters(:,i),fs,fs);
    acts = imfilter(im,filter);
    fea(:,:,i) = acts(hfs+1:size(acts,1)-hfs,hfs+1:size(acts,2)-hfs);
end