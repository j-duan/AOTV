function img = rescale_intensity(img, thres)
p = prctile(img(:), thres);
img(img < p(1)) = p(1);
img(img > p(2)) = p(2);
img = ( img - p(1) ) ./ ( p(2) - p(1) );