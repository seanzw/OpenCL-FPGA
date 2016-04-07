close all;

[batchResult, pipelineResult] = resultParser('kernel/result_lenet5.xml');
batchResult = batchResult / 1e9;
batchResult = batchResult - min(batchResult(:));
axisLimit = [-inf, max(max(batchResult(1 : 2, :, 4))), -inf, inf];

fid = process('kernel/result_lenet5.xml', 1, axisLimit);
fid = process('kernel/result_lenet5_mcu.xml', fid, axisLimit);
