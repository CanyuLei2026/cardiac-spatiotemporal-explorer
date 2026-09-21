function export_case_for_web(input_file, output_dir, sample_count)
% Export one MATLAB v7.3 cardiac simulation case to compact browser assets.
if nargin < 3, sample_count = 2500; end
if ~isfolder(output_dir), mkdir(output_dir); end
source = matfile(input_file);
points_all = source.P_myo;
signals_all = source.U_heart;
time = source.time;
components_all = source.componentID;
point_types_all = source.pointType_myo;
regions_all = source.surfaceRegion_myo;
original_count = size(points_all, 1);
sample_count = min(sample_count, original_count);
rng(7400, 'twister');
sample_indices = sort(randperm(original_count, sample_count));
points = single(points_all(sample_indices, :));
signals = single(signals_all(sample_indices, :));
components = int16(components_all(sample_indices, :));
point_types = int16(point_types_all(sample_indices, :));
regions = int16(regions_all(sample_indices, :));
original_indices = uint32(sample_indices(:) - 1);
write_binary(fullfile(output_dir, 'points.f32'), points', 'single');
write_binary(fullfile(output_dir, 'signals.f32'), signals', 'single');
write_binary(fullfile(output_dir, 'components.i16'), components, 'int16');
write_binary(fullfile(output_dir, 'point_types.i16'), point_types, 'int16');
write_binary(fullfile(output_dir, 'regions.i16'), regions, 'int16');
write_binary(fullfile(output_dir, 'original_indices.u32'), original_indices, 'uint32');
[~, case_name] = fileparts(input_file);
metadata = struct('caseId', case_name, ...
    'pointCount', sample_count, 'originalPointCount', original_count, 'timeCount', numel(time), ...
    'time', double(time(:))', 'coordinateMin', double(min(points, [], 1)), ...
    'coordinateMax', double(max(points, [], 1)), 'signalMin', double(min(signals, [], 'all')), ...
    'signalMax', double(max(signals, [], 'all')), 'components', double(unique(components(:)))', ...
    'pointTypes', double(unique(point_types(:)))', 'regions', double(unique(regions(:)))', ...
    'signalLayout', 'point-major', 'coordinateLayout', 'xyz-interleaved', 'sampleSeed', 7400);
fid = fopen(fullfile(output_dir, 'metadata.json'), 'w');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(metadata, PrettyPrint=true), 'char');
end

function write_binary(path, values, precision)
fid = fopen(path, 'w', 'ieee-le');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, values, precision);
end
