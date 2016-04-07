function drawTimeline(result, fid, color, axisLimit)

    % Get the queue time.
    queueTime = result(:, :, 1);
    
    % Get the submit time.
    submitTime = result(:, :, 2);
    startTime = result(:, :, 3);
    
    
    % Get the end time.
    endTime = result(:, :, 4);
    
    figure(fid);
    hold on;
    axis(axisLimit);
    for i = 1 : size(result, 1)
        for layer = 1 : size(result, 2)
            plot([startTime(i, layer), endTime(i, layer)], [queueTime(i, layer), queueTime(i, layer)], 'Color', color(layer, :), 'LineWidth', 5);
        end
    end
    set(gca,'FontSize',20)
    xlabel('Start/End Time(s)', 'FontSize', 24);
    ylabel('Enqueued Time(s)', 'FontSize', 24);
    legend('WT', 'C1', 'S2', 'C3', 'S4', 'C5', 'F6', 'O7', 'RD');
    hold off;
end