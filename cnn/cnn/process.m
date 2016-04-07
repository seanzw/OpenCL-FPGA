function fid = process(fn, fid)

    [batchResult, pipelineResult] = resultParser(fn);
    batchResult = batchResult - 1e9;
    pipelineResult = pipelineResult - 1e9;
    batchResult = batchResult - min(batchResult(:));
    pipelineResult = pipelineResult - min(pipelineResult(:));

    %% Draw the time line.
    color = rand([size(pipelineResult, 2), 3]);
    drawTimeline(batchResult(1 : 10, :, :), fid, color);
    fid = fid + 1;
    drawTimeline(pipelineResult(1 : 10, :, :), fid, color);
    fid = fid + 1;

    %% Draw the average time for each layer.
    drawLayerTime(batchResult, fid);
    fid = fid + 1;
    drawLayerTime(pipelineResult, fid);
    fid = fid + 1;
end