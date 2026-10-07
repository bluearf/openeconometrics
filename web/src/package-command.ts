const maximumCommandLength = 64_000;

/** Route one installer command through the existing Python execution path. */
export function normalizePackageCommand(command: string): string {
  if (typeof command !== "string" || !command.trim())
    throw new Error("Enter an installation command.");
  if (command.length > maximumCommandLength)
    throw new Error("The command can contain up to 64,000 characters.");
  if (/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/u.test(command))
    throw new Error("Enter the installation command on a single line.");

  const source = command.trim();
  const match =
    /^([%!]?)(?:pip +install|uv +(?:pip +install|add))(?: +(.*))?$/.exec(
      source,
    );
  if (!match) throw new Error("Enter pip install, uv pip install or uv add.");
  if (!match[2])
    throw new Error("Specify a package name or requirements file.");

  const normalized = match[1] ? source : `%${source}`;
  if (normalized.length > maximumCommandLength)
    throw new Error("The command can contain up to 64,000 characters.");
  return normalized;
}
