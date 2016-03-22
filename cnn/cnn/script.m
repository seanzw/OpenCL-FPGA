fid = 1;
result = resultParser('kernel/result_lenet5.xml') / 1e9;
result = result - min(result(:));


%% Draw the time line.
color = rand([size(result, 2), 3]);
drawTimeline(result(1 : 10, :, :), fid, color);
fid = fid + 1;
result = result - min(result(:));

duration = zeros(size(result, 1), size(result, 2), 3);
duration(:, :, 1) = result(:, :, 2) - result(:, :, 1);
duration(:, :, 2) = result(:, :, 3) - result(:, :, 2);
duration(:, :, 3) = result(:, :, 4) - result(:, :, 2);

% Execution time.
sum1 = mean(duration, 1);
sum1 = reshape(sum1(:), [size(result, 2), 3]);
figure(fid);
bar(sum1(:, 3));