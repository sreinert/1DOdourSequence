function [DAQdata, filenames] = load_labview_daq(files_dir, nbuffer, channels_name)
    % LOAD_LABVIEW_DAQ load analog channels recorded with Labview and DAQmx
    %
    % [DAQdata, filenames] = load_labview_daq(files_dir, nbuffer, channels_name)
    %
    % INPUTS
    %   files_dir - directory containing binary files, as a string
    %   nbuffer - number of samples per read (i.e. size of your buffer)
    %   channels_name - (optional) default: (see below)
    %       name of channels, as a cellarray of strings, default being
    %       {'frame_pulse', 'led_driver', 'photodiode', 'stim_id',
    %        'speed_tunnel', 'lick', 'position_tunnel', 'punish', 'reward'}
    %
    % OUTPUTS
    %   DAQdata - extracted times series, as a table with following columns
    %       - 'iterations': Labview iteration counter
    %       - 'labview_time': Labview timer, in seconds
    %       - 'labview_date': date deduced from Labview timer
    %       - 'system_time': system (Windows) timer, in milliseconds
    %       - 'encoder': angular encoder count
    %       - 'speed': speed computed from encoder count
    %       - and 1 column per analog channel found
    %   filenames - file used for each column, as a cellarray of strings
    %
    % REMARKS
    %   All time related columns ('iterations', 'system_time', 'labview_time',
    %   'labview_date') are linearly upscaled by 'nbuffer' to match the number
    %   of samples of analog channels.
    %
    %   The 'iterations' counter starts from 0.
    %
    %   The 'system_time' clock corresponds, on Windows, to the time in
    %   milliseconds from the boot of the machine.
    %
    %   Unnamed channels default to 'achan<chan_number>', where <chan_number>
    %   stands for the last number found in the corresponding filename.
    %
    %   This code is NOT compatible with Matlab earlier than 2016b.
    %
    % SEE ALSO decimate_daqdata

    if ~exist('files_dir', 'var')
        error('Missing files_dir argument.');
    elseif ~isdir(files_dir)
        error('Expected files_dir to be a directory.');
    end

    if ~exist('nbuffer', 'var')
        error('Missing nbuffer argument.');
    end
    nbuffer_attr = {'scalar', 'integer', 'positive'};
    validateattributes(nbuffer, {'numeric'}, nbuffer_attr, '', 'nbuffer');

    % default channel names
    if ~exist('channels_name', 'var')
        channels_name = {
            'frame_pulse', ...
            'led_driver', ...
            'photodiode', ...
            'stim_id', ...
            'speed_tunnel', ...
            'lick', ...
            'position_tunnel', ...
            'punish', ...
            'reward'
        };
    end

    validateattributes(channels_name, {'cell'}, {}, '', 'channels_name');
    for ii = 1:numel(channels_name)
        varname = sprintf('channels_name{%d}', ii);
        validateattributes(channels_name{ii}, {'char'}, {}, varname);
    end

    % find channel files and complain if too few/too many matching
    filename_time = find_files(files_dir, '*_ts.bin', true);
    filename_xraw = find_files(files_dir, '*_xraw.bin', true);
    filename_chans = find_files(files_dir, '*_achn*.bin', false);

    filenames = [repmat({filename_time}, 1, 4), ...
                 repmat({filename_xraw}, 1, 2), ...
                 filename_chans];

    % extract clocks
    DAQdata = extract_time(filename_time, nbuffer);

    % extract encoder position
    fid = fopen(filename_xraw, 'r');
    encoder = fread(fid, inf, 'double', 0, 'ieee-be');
    fclose(fid);

    % remove peaks (usually at the beginning)
    % TODO add warning about this
    encoder(encoder > 4 * 10^9) = 0;
    DAQdata.encoder = encoder;

    % difference from previous tick count, times wheel dimensions, ends up in [m/s]
    % TODO add inputs for wheel parameters instead of hard coding it
    DAQdata.speed = [0; (diff(DAQdata.encoder) * 0.0157) / (100 / 1000)];

    % create channel column names
    chans_name = map_channel_names(channels_name, filename_chans);

    % extract other analog channels
    for ch_idx = 1:numel(filename_chans)
        filename_chan = filename_chans{ch_idx};

        fid = fopen(filename_chan, 'r');
        samples = fread(fid, inf, 'single', 0, 'ieee-be');
        fclose(fid);

        DAQdata.(chans_name{ch_idx}) = samples;
    end
end

function filenames = find_files(files_dir, files_pattern, only_one)
    % helper function to find files matching a simple pattern

    full_pattern = fullfile(files_dir, files_pattern);
    files_info = dir(full_pattern);

    if isempty(files_info)
        filenames = [];
    else
        filenames = fullfile({files_info.folder}, {files_info.name});
    end

    if only_one
        % check that one and only one file has been found
        if isempty(filenames)
            error('No file matching %s!\n', full_pattern);
        elseif numel(filenames) > 1
            error('More than one file matching %s!\n', full_pattern);
        end
        filenames = filenames{1};
    end
end

function chans_name = map_channel_names(channels_name, filename_chans)
    % helper function to convert channel filename into channel names

    chans_tokens = regexp(filename_chans, '_achn(\w+)\.bin', 'tokens');
    chans_num = cellfun(@(x) str2double(x{end}{end}), chans_tokens) + 1;

    n_named_chans = numel(channels_name);
    n_chans = numel(filename_chans);

    chans_name = cell(1, n_chans);
    chans_mask = (chans_num >= 1) & (chans_num <= n_named_chans);
    chans_name(chans_mask) = channels_name(chans_num(chans_mask));
    chans_name(~chans_mask) = arrayfun( ...
        @(x) sprintf('achn%d', x), chans_num(~chans_mask) - 1, 'un', false);
end

function x = linear_upscale(x, n)
    % upscale a column vector with simple linear interpolation
    dx = diff(x)';
    dx = [dx(1), dx];
    incr = linspace(1/n - 1, 0, n)';
    x = reshape(x' + incr .* dx, [], 1);
end

function DAQclocks = extract_time(filename, nbuffer)
    % extract multiple time information

    % structure of BIN file:
    % (i64) seconds since the epoch 01/01/1904 00:00:00.00 UTC (using Gregorian
    %       calendar and ignoring leap seconds)
    % (u64) positive fractions of a second
    % (f64) millisecond timer
    % (f64) iteration number
    fields = {'int64', 'uint64', 'float64', 'float64'};

    fields_size = cellfun(@precision_length, fields);
    sample_size = sum(fields_size);

    % load each time field
    n_fields = numel(fields);
    samples = cell(1, n_fields);

    fid = fopen(filename, 'r');
    for ii = 1:n_fields
        offset = sum(fields_size(1:ii-1));
        skip = sample_size - fields_size(ii);
        fseek(fid, offset, 'bof');
        samples{ii} = fread(fid, inf, fields{ii}, skip, 'ieee-be');
    end
    fclose(fid);

    % assign returned variables
    iterations = linear_upscale(samples{4}, nbuffer);
    system_time = linear_upscale(samples{3}, nbuffer);

    % TODO check division is reasonable (should be 2^64-1?)
    labview_time = samples{1} + samples{2} ./ 2^64;
    labview_time = linear_upscale(labview_time, nbuffer);
    labview_date = datetime([1904, 1, 1]) + seconds(labview_time);
    labview_time = labview_time - labview_time(1);

    DAQclocks = table(iterations, system_time, labview_time, labview_date);
    DAQclocks.Properties.VariableUnits{'labview_time'} = 's';
    DAQclocks.Properties.VariableDescriptions{'labview_time'} = ...
        'recorded with labview timer';
    DAQclocks.Properties.VariableUnits{'system_time'} = 'ms';
    DAQclocks.Properties.VariableDescriptions{'system_time'} = ...
        'recorded with system timer (since boot on Windows)';
end

function psize = precision_length(precision)
    % convert precision format into number of bytes
    switch precision
        case 'single'
            psize = 4;
        case 'double'
            psize = 8;
        case 'int64'
            psize = 8;
        case 'uint64'
            psize = 8;
        case 'float64'
            psize = 8;
        otherwise
            error('conversion of %s format not implemented\n', precision);
    end
end
