function show_error(warped_source, target, padNum, t)

[m,n] = size(target);

abs_error = abs(warped_source-target);

abs_error = abs_error(padNum+1:m-padNum,padNum+1:n-padNum);

% save('difference','abs_error')

figure; imagesc(abs_error); colormap(gray); title('Error After'); axis off; axis equal;

saveas(gcf,[num2str(t),'error_after.png'])