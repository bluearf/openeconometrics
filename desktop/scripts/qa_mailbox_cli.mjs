import { readFile, writeFile } from 'node:fs/promises';
import { createMailboxKey, waitMailboxPacket } from './qa_mailbox.mjs';
import { validateQaManifest } from './verify_windows_cloud_ui.mjs';
const [mode, privatePath, publicPath, argument, source] = process.argv.slice(2);
try {
  if (mode === 'key') await createMailboxKey(privatePath, publicPath, argument, source);
  else if (mode === 'bootstrap') {
    const metadata = JSON.parse(await readFile(publicPath,'utf8'));
    const payload = await waitMailboxPacket(metadata, privatePath, {run_id:metadata.run_id,nonce:metadata.nonce,challenge:'bootstrap',kind:'bootstrap',role:'fixture'});
    const projectId = payload.fixture?.projects?.[0];
    validateQaManifest(payload.fixture, {'project-id':projectId});
    if (payload.fixture.projects.length !== 1 || !/^openecon-qa-[a-z0-9-]{1,90}$/.test(payload.fixture.project_name ?? '')) throw new Error();
    await writeFile(argument,JSON.stringify(payload.fixture)+'\n',{flag:'wx',mode:0o600});
  } else throw new Error();
  process.stdout.write(JSON.stringify({status:'ready',mode})+'\n');
} catch { process.stderr.write('Owned encrypted QA setup failed; no credential was retained in output.\n'); process.exitCode=1; }
