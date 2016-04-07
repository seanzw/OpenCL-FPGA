function fid = process(fn, fid, axisLimit)

    [batchResult, pipelineResult] = resultParser(fn);
    batchResult = batchResult / 1e9;
    pipelineResult = pipelineResult / 1e9;
    batchResult = batchResult - min(batchResult(:));
    pipelineResult = pipelineResult - min(pipelineResult(:));

    %% Draw the time line.
    color = [...
        127, 0, 0;...
        255, 0, 0;...
        255, 127, 0;...
        255, 255, 0;...
        0, 255, 0;...
        0, 0, 255;...
        75, 0, 130;...
        127, 0, 255;...
        0, 0, 0] / 255;
%     axisLimit = [-inf, max(max(batchResult(1 : 2, :, 4))), -inf, inf];
    drawTimeline(batchResult(1 : 2, :, :), fid, color, axisLimit);
    fid = fid + 1;
    drawTimeline(pipelineResult(1 : 2, :, :), fid, color, axisLimit);
    fid = fid + 1;

    %% Draw the average time for each layer.
    drawLayerTime(batchResult, fid);
    fid = fid + 1;
    drawLayerTime(pipelineResult, fid);
    fid = fid + 1;
end