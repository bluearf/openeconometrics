import katex from "katex";
import "katex/dist/katex.min.css";
import { latexMathOptions, validateMathSource } from "./latex-math";

export function renderLatexMath(element: HTMLElement, source: string): void {
  validateMathSource(source);
  katex.render(source, element, latexMathOptions());
}
