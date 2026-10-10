% Development-only exact original-author posterior execution under GNU Octave.
% No alteration of prior_NC.m or forecast_BVAR_NCP.m; no native sampler calls.
args = argv();
assert(numel(args) == 3);
source_root = canonicalize_file_name(args{1});
fixture_path = canonicalize_file_name(args{2});
output_root = canonicalize_file_name(args{3});
assert(~isempty(source_root) && ~isempty(fixture_path) && ~isempty(output_root));
fixture = jsondecode(fileread(fixture_path));
assert(strcmp(fixture.schema, 'openecon.chan2020.original-posterior-fixture.v1'));
assert(isequal(fixture.series_counts(:)', [2 3 4]));
assert(fixture.lags == 4 && fixture.intercept == true);
assert(isequal(size(fixture.levels), [68 4]));
assert(all(isfinite(fixture.levels(:))));
addpath(source_root, '-begin');
assert(strcmp(canonicalize_file_name(which('prior_NC')), fullfile(source_root, 'prior_NC.m')));
original_script = fullfile(source_root, 'forecast_BVAR_NCP.m');
% Queries are made after interpreter startup. No assertion of startup RNG purity.
generator_names = {'rand', 'randn', 'rande', 'randg', 'randp'};
for n = [2 3 4]
    p = 4; k = 1 + n*p;
    Y0 = fixture.levels(1:p, 1:n);
    shortYt = fixture.levels(p+1:end, 1:n);
    Tt = rows(shortYt);
    c1 = fixture.c1; c2 = fixture.c2;
    nsims = 0; burnin = 0;
    before_states = cell(1, numel(generator_names));
    for j = 1:numel(generator_names)
        before_states{j} = feval(generator_names{j}, 'state');
    end
    % The complete unchanged published script executes. 1:0 is an empty loop.
    run(original_script);
    after_states = cell(1, numel(generator_names));
    for j = 1:numel(generator_names)
        after_states{j} = feval(generator_names{j}, 'state');
        assert(isequal(before_states{j}, after_states{j}));
    end
    assert(nsims == 0 && burnin == 0 && isempty(tmpyhat0) && isempty(tmpyhat1));
    % An explicit wrapper-derived inverse; not an extra published endpoint.
    posterior_row_scale = full(KA \ eye(k));
    native_runtime = struct('version', OCTAVE_VERSION, 'computer', computer(), ...
        'original_script', original_script, 'original_prior', which('prior_NC'), ...
        'sampling_iterations', 0, 'startup_RNG_purity_claimed', false);
    output_path = fullfile(output_root, sprintf('original-author-series-%d.mat', n));
    assert(exist(output_path, 'file') == 0);
    save('-mat7-binary', output_path, 'n', 'p', 'k', 'Tt', 'Y0', 'shortYt', ...
        'c1', 'c2', 'nsims', 'burnin', 'A0', 'VA0', 'nu0', 'S0', 'X', 'XX', ...
        'KA', 'Ahat', 'Shat', 'nuhat', 'posterior_row_scale', ...
        'tmpyhat0', 'tmpyhat1', 'generator_names', 'before_states', 'after_states', ...
        'native_runtime');
end
