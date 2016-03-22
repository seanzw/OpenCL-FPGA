function drawTimeline(result, fid, color)

    % Get the queue time.
    queueTime = result(:, :, 1);
    
    % Get the submit time.
    submitTime = result(:, :, 2);
    
    % Get the end time.
    endTime = result(:, :, 4);
    
    figure(fid);
    hold on;
    for i = 1 : size(result, 1)
        for layer = 1 : size(result, 2)
            plot([submitTime(i, layer), endTime(i, layer)], [queueTime(i, layer), queueTime(i, layer)], 'Color', color(layer, :));
        end
    end
    hold off;
end