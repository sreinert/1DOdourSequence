function DAQdata = decimate_daqdata(DAQdata, nz, threshold)
    % DECIMATE_DAQDATA average time series over each frame
    %
    % DAQdata = decimate_daqdata(DAQdata, nz, threshold)
    %
    % This function decimate time series based on start/stop signals of frame
    % acquisition provided by ScanImage. Within each time bin associated with a
    % frame, input time series data are averaged.
    %
    % INPUTS
    %   DAQdata - time series, as a table with at least a 'frame_pulse' column
    %   nz - number of z-planes used during acquisition
    %   threshold - (optional) default: 2.5
    %       threshold used to distinguish when frames were acquired using frame
    %       pulses time serie
    %
    % OUTOUT
    %   DAQdata - decimated time series, as a table with same columns as input
    %
    % REMARKS
    %   Averaging input data within each found time bin is a crude form of
    %   low-pass filter. It should prevent some aliasing but this might not be
    %   enough, depending on the high frequency content of input time series.
    %
    %   For multiple z-planes acquisition, if the number of frames is not a
    %   multiple of 'nz', the last time bin corresponding to an "incomplete"
    %   acquisition is not computed.
    %
    % SEE ALSO load_labview_daq

    % TODO add some low-pass filter?
    % TODO check and drop last frame if seems incomplete (too short...)?
    % TODO warning if drop last set of frames?

    if ~exist('DAQdata', 'var')
        error('Missing DAQdata argument.');
    elseif ~istable(DAQdata)
        error('Expected DAQdata to be a table.');
    elseif ~ismember('frame_pulse', DAQdata.Properties.VariableNames)
        error('Expected DAQdata table to have a frame_pulse column.');
    end

    if ~exist('nz', 'var')
        error('Missing nz argument.');
    end
    nz_attr = {'scalar', 'integer', 'positive'};
    validateattributes(nz, {'numeric'}, nz_attr, '', 'nz');

    if ~exist('threshold', 'var') || isempty(threshold)
        threshold = 2.5;
    end
    validateattributes(threshold, {'numeric'}, {'scalar'}, '', 'threshold');

    % find start/stop of frames
    frame_pulses = diff(DAQdata.frame_pulse > threshold);
    start_idx = find(frame_pulses == 1) + 1;
    stop_idx = find(frame_pulses == -1);
    
    % aggregate time-points of frames from different z-planes
    start_idx = start_idx(1:nz:end);
    stop_idx = stop_idx(nz:nz:end);
    start_idx = start_idx(1:numel(stop_idx));

    % average time series within each frame
    colnames = DAQdata.Properties.VariableNames;
    DAQdata = varfun(@(x) resample_col(x, start_idx, stop_idx), DAQdata);
    DAQdata.Properties.VariableNames = colnames;
end

function col_data = resample_col(col_data, start_idx, stop_idx)
    % helper function to resample column data, dealing with datetime
    if isdatetime(col_data)
        col_data = arrayfun( ...
            @(x,y) mean(col_data(x:y)), start_idx, stop_idx, 'un', false);
        col_data = cat(1, col_data{:});
    else
        col_data = arrayfun(@(x,y) mean(col_data(x:y)), start_idx, stop_idx);
    end
end