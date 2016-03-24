function drawLayerTime(result, fid)
    duration = zeros(size(result, 1), size(result, 2), 3);
    duration(:, :, 1) = result(:, :, 2) - result(:, :, 1);
    duration(:, :, 2) = result(:, :, 3) - result(:, :, 2);
    duration(:, :, 3) = result(:, :, 4) - result(:, :, 3);

    % Execution time.
    sum1 = mean(duration, 1);
    sum1 = reshape(sum1(:), [size(result, 2), 3]);
    figure(fid);
    bar(sum1(:, 3));
end