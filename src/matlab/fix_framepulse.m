function [clean_frame_pulse, false_pulses_idx] = fix_framepulse(bad_frame_pulse, threshold)
    % FIX_FRAMEPULSE try to clean a contaminated frame pulse trace
    %
    % clean_frame_pulse = FIX_FRAMEPULSE(bad_frame_pulse, threshold)
    %
    % This function tries to recover lost frame pulses in a trace that have been
    % contaminated through channel cross-talk. Lost frame pulses are
    % approximately reconstructed by filling gaps between found pulses.
    %
    % INPUTS
    %   bad_frame_pulse - contaminated frame pulses trace, as a vector
    %   threshold - (optional) default: 2.5
    %       threshold used to distinguish when frames were acquired using frame
    %       pulses time serie
    %
    % OUTPUTS
    %   clean_frame_pulse - decontaminated frame pulse trace, as a vector
    %   false_pulses_idx - indices of generated pulses to fill gaps, as a vector
    %
    % REMARKS
    %   This function assumes that the acquisition was uninterrupted from start
    %   to end. If you have gaps in your acquisition, this function will fill
    %   them.
    %
    %   In addition, it assumes that contamination created mostly positive
    %   deflections, and possibly small negative deflections. If negative
    %   deflections makes the signal lower than 'threshold', false positive will
    %   be generated and you will likely get warnings.
    %   Lowering the threshold might help, at the expense of detecting less true
    %   pulses and generating more reconstructed frame pulses.
    %
    % SEE ALSO decimate_daqdata, load_labview_daq

    if ~exist('bad_frame_pulse', 'var')
        error('Missing  bad_frame_pulse argument.');
    end
    validateattributes( ...
        bad_frame_pulse, {'numeric'}, {'vector'}, '', 'bad_frame_pulse');

    if ~exist('threshold', 'var') || isempty(threshold)
        threshold = 2.5;
    end
    validateattributes(threshold, {'numeric'}, {'scalar'}, '', 'threshold');

    % isolate core part of frame pulse trace
    idx_start = find(bad_frame_pulse > threshold, 1);
    idx_stop = find(bad_frame_pulse > threshold, 1, 'last');

    % find frame pulses and estimate their period
    frame_pulses = diff(bad_frame_pulse > threshold);
    pulse_idx = find(frame_pulses == 1);
    delta_frames = mean(diff(pulse_idx));

    % reconstruct frames pulses trace with the ones found
    clean_frame_pulse = zeros(size(bad_frame_pulse)) + 5;
    clean_frame_pulse(1:idx_start) = bad_frame_pulse(1:idx_start);
    clean_frame_pulse(idx_stop+1:end) = bad_frame_pulse(idx_stop+1:end);
    clean_frame_pulse(pulse_idx) = 0;

    % find gaps in frame pulses and fill them approximately
    false_pulses_idx = [];

    for ii = 1:numel(pulse_idx)-1
        delta_pulses = pulse_idx(ii + 1) - pulse_idx(ii);

        if delta_pulses < delta_frames - 1
            warning('fix_framepulse:earlyFrame', ...
                ['A frame pulse have been detected too early, at frame %d. ' ...
                 'Check the documentation for advices.'], pulse_idx(ii + 1));
        end

        if delta_pulses < delta_frames + 1
            continue;
        end

        n_pulses = round(delta_pulses / delta_frames) + 1;
        extra_pulses = linspace(pulse_idx(ii), pulse_idx(ii + 1), n_pulses);
        extra_pulses = round(extra_pulses(2:end-1));

        clean_frame_pulse(extra_pulses) = 0;
        false_pulses_idx = [false_pulses_idx, extra_pulses];
    end
end