import assert from "node:assert/strict";
import test from "node:test";
import { normalizePackageCommand } from "../src/package-command.ts";

test("bare pip and uv commands use the workspace installer syntax", () => {
  assert.equal(
    normalizePackageCommand("pip install humanize polars==1.44.2"),
    "%pip install humanize polars==1.44.2",
  );
  assert.equal(
    normalizePackageCommand("uv pip install humanize"),
    "%uv pip install humanize",
  );
  assert.equal(normalizePackageCommand("uv add humanize"), "%uv add humanize");
});

test("explicit magic and bang prefixes remain unchanged", () => {
  for (const prefix of [
    "%pip install",
    "!pip install",
    "%uv pip install",
    "!uv pip install",
    "%uv add",
    "!uv add",
  ]) {
    const command = `${prefix} humanize==4.16.0`;
    assert.equal(normalizePackageCommand(command), command);
  }
});

test("quoted extras, ranges, markers and requirements paths retain exact argument text", () => {
  const commands = [
    `pip install 'demo[fast]>=1,<2; python_version >= "3.11"' another==2.0`,
    `uv pip install --upgrade -r "/Users/örnek/Project files/requirements.txt"`,
    `uv add -U --requirement='requirements file.txt'`,
    `pip install -rrequirements.txt # project dependencies`,
    `uv  pip   install  "humanize>=4,<5"`,
  ];
  for (const command of commands)
    assert.equal(normalizePackageCommand(command), `%${command}`);
});

test("only surrounding whitespace is removed; quoted whitespace is preserved", () => {
  assert.equal(
    normalizePackageCommand(`  uv pip install -r 'my  requirements.txt'   `),
    `%uv pip install -r 'my  requirements.txt'`,
  );
});

test("empty commands and missing installer arguments are rejected", () => {
  for (const command of [
    "",
    "   ",
    "pip install",
    "%pip install ",
    "!pip install",
    "uv pip install",
    "%uv add",
  ])
    assert.throws(() => normalizePackageCommand(command), Error);
});

test("ordinary Python and unrelated terminal commands cannot become submitted code", () => {
  for (const command of [
    "print('hello')",
    "oe.install('humanize')",
    "import os; os.system('echo bad')",
    "python -m pip install humanize",
    "sudo pip install humanize",
    "pip uninstall humanize",
    "pip list",
    "uv run script.py",
    "uv sync",
    "uv install humanize",
    "% pip install humanize",
    "pip installer humanize",
    "uv additional humanize",
    "PIP install humanize",
    "%pipinstall humanize",
  ])
    assert.throws(() => normalizePackageCommand(command), Error);
});

test("multiline input and every ASCII control character are rejected", () => {
  for (const command of [
    "pip install humanize\nprint('must not execute')",
    "%pip install humanize\r\n%uv add polars",
    "pip install humanize\n",
    "\npip install humanize",
    "pip install humanize\u0085print('no')",
    "pip install humanize\u2028print('no')",
    "pip install humanize\u2029print('no')",
  ])
    assert.throws(() => normalizePackageCommand(command), Error);
  for (const code of [...Array(32).keys(), 127])
    assert.throws(
      () =>
        normalizePackageCommand(
          `pip install humanize${String.fromCharCode(code)}`,
        ),
      Error,
    );
});

test("bounds apply to the generated Python source, including the inserted magic prefix", () => {
  const explicitPrefix = "%pip install ";
  const explicit = explicitPrefix + "x".repeat(64_000 - explicitPrefix.length);
  assert.equal(normalizePackageCommand(explicit).length, 64_000);
  const bare = explicit.slice(1);
  assert.equal(normalizePackageCommand(bare), explicit);
  assert.throws(() => normalizePackageCommand(explicit + "x"), /64,000/);
  assert.throws(() => normalizePackageCommand(bare + "x"), /64,000/);
});

test("requirements and option validation stays with the existing Python installer", () => {
  const command = `pip install 'demo[fast]>=1,<2; sys_platform == "darwin"' --upgrade`;
  assert.equal(normalizePackageCommand(command), `%${command}`);
});
