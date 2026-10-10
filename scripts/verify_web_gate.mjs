// A Node test process exiting zero with no tests or skipped tests is not a gate.
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
const deadline = performance.now() + 600_000;
const files = readdirSync('web/tests').filter(x => x.endsWith('.test.ts')).map(x => `web/tests/${x}`);
if (!files.length) throw new Error('No web tests discovered');

function runSuite(selected, expectedTests) {
  const timeout = Math.floor(deadline - performance.now());
  if (timeout <= 0) throw new Error('Node test gate exceeded its shared 600-second deadline');
  const result = spawnSync(process.execPath, ['--experimental-strip-types', '--test', '--test-reporter=tap', ...selected],
    { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout });
  process.stdout.write(result.stdout ?? '');
  process.stderr.write(result.stderr ?? '');
  if (result.error || result.status !== 0) process.exit(1);
  const counts = Object.fromEntries([...result.stdout.matchAll(/^# (tests|pass|fail|cancelled|skipped|todo) (\d+)$/gm)]
    .map(x => [x[1], Number(x[2])]));
  if (!(counts.tests > 0 && (expectedTests === undefined || counts.tests === expectedTests) &&
        counts.pass === counts.tests && counts.fail === 0 && counts.cancelled === 0 &&
        counts.skipped === 0 && counts.todo === 0 && performance.now() <= deadline)) {
    console.error('Incomplete test gate', { selected, counts, expectedTests });
    process.exit(1);
  }
}

runSuite(files);
// Node counts an empty file as one passing test; require every Cloud guard separately.
runSuite(['desktop/scripts/verify_windows_cloud_ui.test.mjs'], 21);
