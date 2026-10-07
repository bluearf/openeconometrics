import type { KatexOptions } from "katex";

export const MAX_MATH_LENGTH = 65_536;

export function latexMathOptions(): KatexOptions {
  return {
    displayMode: true,
    output: "htmlAndMathml",
    throwOnError: true,
    errorColor: "#14263d",
    trust: false,
    strict: (code) => (code === "unicodeTextInMathMode" ? "ignore" : "error"),
    maxExpand: 1_000,
    maxSize: 20,
    macros: {},
  };
}

export function validateMathSource(source: string): void {
  if (source.length > MAX_MATH_LENGTH)
    throw new Error("The LaTeX preview limit was exceeded.");
}
