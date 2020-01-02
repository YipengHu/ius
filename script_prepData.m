% script_prepData

DataSetFolder = fullfile(getenv('HOME'), 'Scratch/data/protocol/SPE_data_classes');

ClassFolders = { 
    [DataSetFolder, '/1_skull']; 
    [DataSetFolder, '/2_abdomen'];
    [DataSetFolder, '/3_heart'];
    [DataSetFolder, '/4_other']; };

% go through all cases
case_ids = {};
frame_counters = zeros(length(ClassFolders),1);
frame_filenames = cell(length(ClassFolders),1);
for i = 1:length(ClassFolders)
    frame_names = dir(ClassFolders{i});
    for j = 3:length(frame_names)
        fname = frame_names(j).name;
        % dealing with different format here
        date = strfind(fname,'-');
        if ~isempty(date)
            start0=date(end)+1;
        else
            start0=1;
        end
        ext0 = regexpi(fname,'.png');
        newstr = strrep(fname(start0:ext0-1),'fr_','');
        % additional check here
        udls = strfind(newstr,'_');
        if length(udls) ~= 2
            warning('Incorrect filename format!, %s',fname);
        end
        id = newstr(1:udls(2)-1);
        fr = str2double(newstr(udls(2)+1:end));
        [~, idx] = ismember(id, case_ids);
        if idx  % add to exisiting volume
        else  % create a new one
        end
    end 
end
