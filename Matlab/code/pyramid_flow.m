function [u, v] = pyramid_flow(source, target, levels, talyor, maxIter, lambda, tolerance, difference, mode)

tic
figure(3)
for scale = levels %multi scale
    
    fprintf('registration at sacle %d \n', scale)
    
    source_ = imresize(source, size(source)/scale, 'bilinear');
    target_ = imresize(target, size(target)/scale, 'bilinear');
    
    if scale == levels(1)
        u = zeros(size(source_));
        v = zeros(size(source_));
    else
        u = imresize(u, 2, 'bilinear')*2;
        v = imresize(v, 2, 'bilinear')*2;
    end
        
    clear diff
    for talyor_expension = 1 : talyor %multi linearisations
        fprintf(' mode is %s, talyor expension %d times;', mode, talyor_expension)
        
        warped_source = imwarp(source_, cat(3, u, v), 'Interp', 'linear');
        if strcmp(mode, 'tv')
            [u, v] =   tvl1_optimizer(u, v, warped_source, target_, maxIter, lambda, tolerance);
        elseif strcmp(mode, 'sotv')            
            [u, v] = sotvl1_optimizer(u, v, warped_source, target_, maxIter, lambda, tolerance);
        elseif strcmp(mode, 'totv')
            [u, v] = totvl1_optimizer(u, v, warped_source, target_, maxIter, lambda, tolerance);
        elseif strcmp(mode, 'fotv')
            [u, v] = fotvl1_optimizer(u, v, warped_source, target_, maxIter, lambda, tolerance);
        end

        % convergence check
        diff(talyor_expension) = sum(sum((warped_source-target_).^2));
        if talyor_expension ~= 1
            if abs(diff(end-1)-diff(end))*100/diff(end-1) <= difference
                break
            end
        end
        imagesc(warped_source); colormap(gray); axis off; axis equal; title('frame 1');
        pause(0.0001);
    end
end
close(3)
toc
