function warped_source = show_warped_image(source, u, v, padNum, t, interp_mode)

[m,n] = size(source);

warped_source = imwarp(source, cat(3, u, v), 'Interp', interp_mode); 

figure; imagesc(warped_source(padNum+1:m-padNum,padNum+1:n-padNum)); colormap(gray); title('Warped Source'); axis off; axis equal;

% saveas(gcf,[num2str(t),'warped_source.png'])