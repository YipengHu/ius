% script_prepData
normFolder = fullfile(getenv('HOME'), 'Scratch/data/protocol/normalised');
mkdir(normFolder);
dataFolder = fullfile(getenv('HOME'), 'Scratch/data/protocol/SPE_data_classes');
ClassNames = {'1_skull'; '2_abdomen'; '3_heart'; '4_other'};
ClassFolders = cellfun(@(x)fullfile(dataFolder,x),ClassNames,'UniformOutput',false);

% go through all cases
case_ids = {};
frame_info = {};
frame_counters = zeros(length(ClassFolders),1);
idx_frame_1 = 0;
for idx_class_1 = 1:length(ClassFolders)
    frame_names = dir(ClassFolders{idx_class_1});
    for j = 3:length(frame_names)
        fname = frame_names(j).name;
        % get rid of the problematic files
        try
            % debug: fprintf('reading No.%d - [%s]\n',i,filename)
            img = imread(fullfile(frame_names(j).folder,fname));  % figure, imshow(img,[])
        catch
            disp(fullfile(frame_names(j).folder,fname))
            continue
        end       
        % dealing with different date format here
        date_del = strfind(fname,'-');
        if length(date_del)>=6  % case that date_del is repeated
            start0=date_del(6)+1;
        elseif length(date_del)>=3
            start0=date_del(3)+1;
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
        [~, idx_case_1] = ismember(id, case_ids);
        if idx_case_1==0  % add to exisiting volume
            idx_case_1 = length(case_ids)+1;
            case_ids{length(case_ids)+1} = id;
        end
        idx_frame_1 = idx_frame_1+1;
        idx_frame = idx_frame_1 - 1;
        frame_info(idx_frame_1).filename = fname;
        frame_info(idx_frame_1).case_name = id;
        frame_info(idx_frame_1).case_idx = idx_case_1 - 1;
        frame_info(idx_frame_1).class_name = ClassNames{idx_class_1};
        frame_info(idx_frame_1).class_idx = idx_class_1 - 1;
    end
end


save(fullfile(normFolder,'frame_info'),'frame_info');

%% now write into files
roi_crop = [47,230,33,288]; % [ymin,ymax,xmin,xmax]
indices_class = [frame_info(:).class_idx];
indices_subject = [frame_info(:).case_idx];

% by frames
h5fn_frames = fullfile(normFolder,'protocol_sweep_class_frames.h5');  delete(h5fn_frames);
for idx_frame_1 = 1:length(frame_info)
    %% now read in image
    filename = fullfile(dataFolder,frame_info(idx_frame_1).class_name,frame_info(idx_frame_1).filename);
    img = imread(filename);  
    img = img(roi_crop(1):roi_crop(2),roi_crop(3):roi_crop(4));
    % figure, imshow(img,[])
    GroupName = sprintf('/frame%08d',idx_frame_1-1);
    h5create(h5fn_frames,GroupName,size(img),'DataType','uint8');
    h5write(h5fn_frames,GroupName,img);
end
GroupName = '/class';
h5create(h5fn_frames,GroupName,size(indices_class),'DataType','uint32');
h5write(h5fn_frames,GroupName,uint32(indices_class));
GroupName = '/subject';
h5create(h5fn_frames,GroupName,size(indices_subject),'DataType','uint32');
h5write(h5fn_frames,GroupName,uint32(indices_subject));

% by subject
h5fn_subjects = fullfile(normFolder,'protocol_sweep_class_subjects.h5');  delete(h5fn_subjects);
for idx_subject = unique(indices_subject)
    frame_subject = 0;
    indices_frame_1_subject = find(indices_subject==idx_subject);
    for idx_frame_1 = indices_frame_1_subject
        filename = fullfile(dataFolder,frame_info(idx_frame_1).class_name,frame_info(idx_frame_1).filename);
        img = imread(filename);
        img = img(roi_crop(1):roi_crop(2),roi_crop(3):roi_crop(4));        
        GroupName = sprintf('/subject%06d_frame%08d',idx_subject,frame_subject);
        frame_subject = frame_subject+1;
        h5create(h5fn_subjects,GroupName,size(img),'DataType','uint8');
        h5write(h5fn_subjects,GroupName,img);
    end
    GroupName = sprintf('/subject%06d_class',idx_subject);
    h5create(h5fn_frames,GroupName,size(indices_frame_1_subject),'DataType','uint32');
    h5write(h5fn_frames,GroupName,uint32(indices_class(indices_frame_1_subject)));
end

