export const PROJECT_CREATE_LIMITS = { name: 100, description: 500 } as const;

export interface ProjectCreateInput {
  name: string;
  description: string;
}

// Python strip() and Unicode code-point lengths define the server contract.
// FEFF is content rather than whitespace; forbidden controls are checked raw.
const trimmed = (value: string) =>
  value.replace(/^\p{White_Space}+|\p{White_Space}+$/gu, "");

export function validateProjectDraft(name: string, description: string) {
  const input = { name: trimmed(name), description: trimmed(description) };
  const lengths = {
    name: Array.from(input.name).length,
    description: Array.from(input.description).length,
  };
  const errors: Partial<Record<keyof ProjectCreateInput, string>> = {};
  if (/[\p{Cc}\p{Cs}\u2028\u2029]/u.test(name))
    errors.name = "Remove control characters from the project name.";
  else if (!lengths.name || lengths.name > PROJECT_CREATE_LIMITS.name)
    errors.name = "The project name must contain 1–100 characters.";
  if (
    Array.from(description).some(
      (character) =>
        /[\p{Cc}\p{Cs}]/u.test(character) && !"\t\r\n".includes(character),
    )
  )
    errors.description =
      "Remove control characters from the description. Tabs and line breaks are allowed.";
  else if (lengths.description > PROJECT_CREATE_LIMITS.description)
    errors.description = "The description can contain up to 500 characters.";
  return { input, lengths, errors };
}
