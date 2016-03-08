function result = resultParser(fn)

    try
        tree = xmlread(fn);
    catch
        error('Failed reading XML file %s.', fn);
    end
    
    inputs = tree.getElementsByTagName('input');
    events = inputs.item(0).getElementsByTagName('event');
    result = zeros(inputs.getLength, events.getLength, 4);
    
    for i = 0 : inputs.getLength - 1
        events = inputs.item(i).getElementsByTagName('event');
        for j = 0 : events.getLength - 1
            result(i + 1, j + 1, 1) = str2double(events.item(j).getElementsByTagName('que').item(0).getFirstChild.getData);
            result(i + 1, j + 1, 2) = str2double(events.item(j).getElementsByTagName('sub').item(0).getFirstChild.getData);
            result(i + 1, j + 1, 3) = str2double(events.item(j).getElementsByTagName('sta').item(0).getFirstChild.getData);
            result(i + 1, j + 1, 4) = str2double(events.item(j).getElementsByTagName('end').item(0).getFirstChild.getData);
        end
    end
end