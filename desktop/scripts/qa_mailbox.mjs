// One-run encrypted QA delivery. No password, OTP, administrator credential or private key is retained in GitHub.
import { createCipheriv, createDecipheriv, generateKeyPairSync, privateDecrypt, publicEncrypt, randomBytes, constants } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';

const repository = 'bluearf/openecon';
const fields = ['run_id','nonce','challenge','kind','role'];
export function validateContext(value) {
  if (!value || Object.keys(value).sort().join() !== [...fields].sort().join()
      || !/^[0-9]{8,20}$/.test(value.run_id) || !/^[a-f0-9]{64}$/.test(value.nonce)
      || !/^(bootstrap|[a-f0-9]{32})$/.test(value.challenge)
      || !['bootstrap','email-code'].includes(value.kind)
      || !['fixture','owner','viewer'].includes(value.role)
      || (value.kind === 'bootstrap') !== (value.role === 'fixture')
      || (value.kind === 'bootstrap') !== (value.challenge === 'bootstrap'))
    throw new Error('Invalid one-run QA delivery context.');
  return Object.fromEntries(fields.map(key => [key,value[key]]));
}
const aad = context => Buffer.from(JSON.stringify(validateContext(context)));
export function seal(publicKey, context, payload) {
  const key = randomBytes(32), iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', key, iv); cipher.setAAD(aad(context));
  const plain = Buffer.from(JSON.stringify(payload));
  if (plain.length > 32768) throw new Error('QA delivery exceeds its bound.');
  const ciphertext = Buffer.concat([cipher.update(plain),cipher.final()]);
  const encryptedKey = publicEncrypt({key:publicKey,oaepHash:'sha256',padding:constants.RSA_PKCS1_OAEP_PADDING},key);
  key.fill(0); plain.fill(0);
  return {schema:1,context:validateContext(context),encrypted_key:encryptedKey.toString('base64'),iv:iv.toString('base64'),tag:cipher.getAuthTag().toString('base64'),ciphertext:ciphertext.toString('base64')};
}
export function unseal(privateKey, context, envelope, now = Date.now()) {
  try {
    if (envelope?.schema !== 1 || JSON.stringify(validateContext(envelope.context)) !== aad(context).toString()) throw new Error();
    for (const field of ['encrypted_key','iv','tag','ciphertext']) {
      if (typeof envelope[field] !== 'string' || envelope[field].length > 65536 || !/^[A-Za-z0-9+/]*={0,2}$/.test(envelope[field])) throw new Error();
    }
    if (Buffer.from(envelope.iv,'base64').length !== 12 || Buffer.from(envelope.tag,'base64').length !== 16) throw new Error();
    const key = privateDecrypt({key:privateKey,oaepHash:'sha256',padding:constants.RSA_PKCS1_OAEP_PADDING},Buffer.from(envelope.encrypted_key,'base64'));
    const decipher = createDecipheriv('aes-256-gcm',key,Buffer.from(envelope.iv,'base64'));
    decipher.setAAD(aad(context));decipher.setAuthTag(Buffer.from(envelope.tag,'base64'));
    const plain = Buffer.concat([decipher.update(Buffer.from(envelope.ciphertext,'base64')),decipher.final()]);key.fill(0);
    if (plain.length > 32768) throw new Error();
    const payload = JSON.parse(plain.toString());plain.fill(0);
    if (!Number.isSafeInteger(payload.expires_at) || payload.expires_at <= now || payload.expires_at > now+20*60*1000) throw new Error();
    if (context.kind === 'email-code' && (!/^[0-9]{6}$/.test(payload.code) || typeof payload.email !== 'string')) throw new Error();
    return payload;
  } catch { throw new Error('QA delivery is invalid, expired or belongs to another challenge.'); }
}
export async function createMailboxKey(privatePath, publicPath, branch, sourceCommit) {
  if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_REPOSITORY !== repository
      || !/^codex\/qa-mailbox-[a-f0-9]{32}$/.test(branch) || !/^[a-f0-9]{40}$/.test(sourceCommit))
    throw new Error('Only an owned hosted QA mailbox is allowed.');
  const {publicKey,privateKey}=generateKeyPairSync('rsa',{modulusLength:3072,publicKeyEncoding:{type:'spki',format:'pem'},privateKeyEncoding:{type:'pkcs8',format:'pem'}});
  const metadata={schema:1,run_id:process.env.GITHUB_RUN_ID,run_attempt:process.env.GITHUB_RUN_ATTEMPT,nonce:randomBytes(32).toString('hex'),mailbox_branch:branch,controller_source:process.env.GITHUB_SHA,installer_source:sourceCommit,public_key:publicKey};
  await writeFile(privatePath,privateKey,{flag:'wx',mode:0o600});
  await writeFile(publicPath,JSON.stringify(metadata)+'\n',{flag:'wx',mode:0o600});
  return metadata;
}
export async function waitMailboxPacket(metadata, privatePath, context, timeoutMs=600000) {
  validateContext(context);
  if (process.env.GITHUB_ACTIONS !== 'true' || process.env.GITHUB_REPOSITORY !== repository
      || metadata.run_id !== process.env.GITHUB_RUN_ID || context.run_id !== metadata.run_id || context.nonce !== metadata.nonce
      || !/^codex\/qa-mailbox-[a-f0-9]{32}$/.test(metadata.mailbox_branch) || !process.env.GITHUB_TOKEN)
    throw new Error('The QA mailbox is not bound to this hosted run.');
  const privateKey=await readFile(privatePath,'utf8');
  const deadline=Date.now()+Math.min(timeoutMs,1200000);
  while(Date.now()<deadline) {
    const response=await fetch(`https://api.github.com/repos/${repository}/contents/packet.json?ref=${encodeURIComponent(metadata.mailbox_branch)}`,{headers:{Authorization:`Bearer ${process.env.GITHUB_TOKEN}`,Accept:'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},signal:AbortSignal.timeout(15000)});
    if(response.ok) {
      const body=await response.json();
      if(body.encoding!=='base64'||typeof body.content!=='string'||body.content.length>131072) throw new Error('Invalid bounded encrypted QA mailbox.');
      const envelope=JSON.parse(Buffer.from(body.content,'base64').toString());
      if(envelope.context?.challenge===context.challenge && envelope.context?.kind===context.kind && envelope.context?.role===context.role)
        return unseal(privateKey,context,envelope);
    } else if(response.status!==404) throw new Error('The encrypted QA mailbox could not be read.');
    await delay(3000);
  }
  throw new Error('The one-run QA delivery timed out.');
}
