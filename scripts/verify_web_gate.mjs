// A Node test process exiting zero with no tests or skipped tests is not a gate.
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
const files = readdirSync('web/tests').filter(x => x.endsWith('.test.ts')).map(x => `web/tests/${x}`);
if (!files.length) throw new Error('No web tests discovered');
const result = spawnSync(process.execPath, ['--experimental-strip-types', '--test', '--test-reporter=tap', ...files],
  { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout: 600_000 });
process.stdout.write(result.stdout ?? '');
process.stderr.write(result.stderr ?? '');
if (result.error || result.status !== 0) process.exit(1);
const counts = Object.fromEntries([...result.stdout.matchAll(/^# (tests|pass|fail|cancelled|skipped|todo) (\d+)$/gm)]
  .map(x => [x[1], Number(x[2])]));
if (!(counts.tests > 0 && counts.pass === counts.tests && counts.fail === 0 &&
      counts.cancelled === 0 && counts.skipped === 0 && counts.todo === 0)) {
  console.error('Incomplete test gate', counts);
  process.exit(1);
}
