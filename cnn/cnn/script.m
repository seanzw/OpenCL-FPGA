result = resultParser('result_lenet5_FPGA.xml');
result = result - min(result(:));

duration = zeros(size(result, 1), size(result, 2), 3);
duration(:, :, 1) = result(:, :, 2) - result(:, :, 1);
duration(:, :, 2) = result(:, :, 3) - result(:, :, 2);
duration(:, :, 3) = result(:, :, 4) - result(:, :, 3);

% Execution time.
sum1 = mean(duration, 1);
sum1 = reshape(sum1(:), [size(result, 2), 3]);
figure(1);
bar(sum1(:, 3));