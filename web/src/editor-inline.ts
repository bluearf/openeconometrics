import { completionStatus } from "@codemirror/autocomplete";
import {
  Facet,
  Prec,
  StateEffect,
  StateField,
  type EditorState,
  type Extension,
} from "@codemirror/state";
import {
  Decoration,
  EditorView,
  ViewPlugin,
  WidgetType,
  keymap,
  type DecorationSet,
  type ViewUpdate,
} from "@codemirror/view";

export interface InlineSuggestionRequest {
  prefix: string;
  suffix: string;
  language: "python";
  signal: AbortSignal;
}

/** The caller owns credentials, transport and permission to share source code. */
export interface InlineSuggestionProvider {
  request(
    request: InlineSuggestionRequest,
  ): Promise<string | null> | string | null;
}

export interface InlineSuggestionOptions {
  provider?: InlineSuggestionProvider;
  /** A project/file identity; changes invalidate all pending suggestions. */
  contextKey?: string;
}

export const INLINE_IDLE_MS = 400;
export const INLINE_TIMEOUT_MS = 2500;
export const INLINE_PREFIX_CHARS = 4096;
export const INLINE_SUFFIX_CHARS = 2048;
export const INLINE_MAX_CHARS = 512;
export const INLINE_MAX_LINES = 3;

export function inlineSuggestionContext(
  state: EditorState,
  pos = state.selection.main.head,
) {
  return {
    prefix: state.sliceDoc(Math.max(0, pos - INLINE_PREFIX_CHARS), pos),
    suffix: state.sliceDoc(
      pos,
      Math.min(state.doc.length, pos + INLINE_SUFFIX_CHARS),
    ),
  };
}

const configuration = Facet.define<
  InlineSuggestionOptions,
  InlineSuggestionOptions
>({
  combine: (values) => values[0] ?? {},
});

interface Suggestion {
  pos: number;
  text: string;
}

const setSuggestion = StateEffect.define<Suggestion | null>();
const dismissSuggestion = StateEffect.define<null>();

function sameConfiguration(
  a: InlineSuggestionOptions,
  b: InlineSuggestionOptions,
) {
  return a.provider === b.provider && a.contextKey === b.contextKey;
}

const suggestionState = StateField.define<Suggestion | null>({
  create: () => null,
  update(value, transaction) {
    if (
      transaction.docChanged ||
      transaction.selection ||
      transaction.state.readOnly ||
      completionStatus(transaction.state) !== null ||
      !sameConfiguration(
        transaction.startState.facet(configuration),
        transaction.state.facet(configuration),
      )
    )
      value = null;
    for (const effect of transaction.effects) {
      if (effect.is(setSuggestion)) value = effect.value;
      if (effect.is(dismissSuggestion)) value = null;
    }
    return value;
  },
});

/** Treat model output as plain continuation text, never HTML or executable work. */
export function normalizeInlineSuggestion(
  raw: unknown,
  prefix: string,
  suffix: string,
): string | null {
  if (typeof raw !== "string" || !raw || raw.length > 8192) return null;
  let text = raw.replace(/\r\n?/g, "\n");
  const fenced = /^\s*```(?:python|py)?\s*\n([\s\S]*?)\n```\s*$/i.exec(text);
  if (fenced) text = fenced[1];
  else if (text.includes("```")) return null;
  const explanation =
    /^\s*(?:here(?:'s| is)|sure[,!:]|certainly[,!:]|the following|you can|this code|işte\b|elbette\b|aşağıda\b)/i;
  if (explanation.test(text) || explanation.test(text.toLocaleLowerCase("tr")))
    return null;
  if (text.includes("\0")) return null;
  const lastLine = prefix.slice(prefix.lastIndexOf("\n") + 1);
  if (prefix && text.startsWith(prefix)) text = text.slice(prefix.length);
  else if (lastLine && text.startsWith(lastLine))
    text = text.slice(lastLine.length);
  const repeatedSuffix = suffix.trimEnd();
  if (repeatedSuffix.trim().length > 1) {
    const repeatedAt = text.indexOf(repeatedSuffix);
    if (repeatedAt >= 0) text = text.slice(0, repeatedAt);
  }
  text = text.trimEnd();
  // Some providers repeat the following line. Keep the existing document once.
  for (
    let length = Math.min(text.length, suffix.length);
    length > 0;
    length--
  ) {
    const overlap = suffix.slice(0, length);
    if (overlap.trim() && text.endsWith(overlap)) {
      text = text.slice(0, -length);
      break;
    }
  }
  if (!text.trim()) return null;
  const lines = text.split("\n");
  if (lines.length > INLINE_MAX_LINES || text.length > INLINE_MAX_CHARS)
    return null;
  return text;
}

class GhostText extends WidgetType {
  constructor(readonly text: string) {
    super();
  }
  eq(other: GhostText) {
    return this.text === other.text;
  }
  get lineBreaks() {
    return this.text.split("\n").length - 1;
  }
  toDOM(view: EditorView) {
    const span = view.dom.ownerDocument.createElement("span");
    span.className = "cm-inline-suggestion";
    span.textContent = this.text;
    span.setAttribute("aria-hidden", "true");
    span.setAttribute("data-inline-suggestion", "true");
    return span;
  }
  ignoreEvent() {
    return true;
  }
}

const inlineController = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet = Decoration.none;
    private idleTimer: ReturnType<typeof setTimeout> | null = null;
    private timeoutTimer: ReturnType<typeof setTimeout> | null = null;
    private controller: AbortController | null = null;
    private generation = 0;
    private alive = true;
    private interacted = false;
    private dismissed = false;
    private hidden = false;
    constructor(private view: EditorView) {
      this.decorate();
    }

    update(update: ViewUpdate) {
      const contextChanged = !sameConfiguration(
        update.startState.facet(configuration),
        update.state.facet(configuration),
      );
      const explicitlyDismissed = update.transactions.some((transaction) =>
        transaction.effects.some((effect) => effect.is(dismissSuggestion)),
      );
      if (
        update.transactions.some((transaction) =>
          transaction.effects.some(
            (effect) => effect.is(setSuggestion) && effect.value !== null,
          ),
        )
      )
        this.hidden = false;
      if (update.docChanged) {
        this.interacted = true;
        this.dismissed = false;
      }
      if (explicitlyDismissed) this.dismissed = true;
      const completionChanged =
        completionStatus(update.startState) !== completionStatus(update.state);
      if (contextChanged || update.state.readOnly) this.interacted = false;
      if (
        update.docChanged ||
        update.selectionSet ||
        update.focusChanged ||
        contextChanged ||
        update.state.readOnly ||
        completionChanged ||
        explicitlyDismissed
      ) {
        this.cancel();
        if (update.focusChanged && !this.view.hasFocus) {
          this.hidden = true;
          const generation = this.generation;
          queueMicrotask(() => {
            if (
              this.alive &&
              this.generation === generation &&
              !this.view.hasFocus
            )
              this.view.dispatch({ effects: setSuggestion.of(null) });
          });
        }
        if (
          !contextChanged &&
          this.interacted &&
          !this.dismissed &&
          this.eligible()
        )
          this.idleTimer = setTimeout(() => {
            this.idleTimer = null;
            this.request();
          }, INLINE_IDLE_MS);
      }
      this.decorate();
    }

    private decorate() {
      const suggestion = this.view.state.field(suggestionState);
      this.decorations =
        suggestion && this.view.hasFocus && !this.hidden
          ? Decoration.set([
              Decoration.widget({
                widget: new GhostText(suggestion.text),
                side: 1,
              }).range(suggestion.pos),
            ])
          : Decoration.none;
    }

    private eligible() {
      const state = this.view.state,
        config = state.facet(configuration),
        selection = state.selection.main;
      if (
        !config.provider ||
        !config.contextKey?.trim() ||
        state.readOnly ||
        !this.view.hasFocus ||
        state.selection.ranges.length !== 1 ||
        !selection.empty ||
        completionStatus(state) !== null
      )
        return false;
      const line = state.doc.lineAt(selection.head);
      // Auto-close brackets and quotes may follow the cursor; identifiers may not.
      return (
        /^[\t )\]}"']*$/.test(state.sliceDoc(selection.head, line.to)) &&
        !!state
          .sliceDoc(
            Math.max(0, selection.head - INLINE_PREFIX_CHARS),
            selection.head,
          )
          .trim()
      );
    }

    private request() {
      if (!this.alive || this.dismissed || !this.eligible()) return;
      const state = this.view.state,
        config = state.facet(configuration),
        pos = state.selection.main.head;
      const { prefix, suffix } = inlineSuggestionContext(state, pos);
      const generation = ++this.generation,
        controller = new AbortController();
      this.controller = controller;
      this.timeoutTimer = setTimeout(() => this.cancel(), INLINE_TIMEOUT_MS);
      Promise.resolve()
        .then(() =>
          controller.signal.aborted
            ? null
            : config.provider!.request({
                prefix,
                suffix,
                language: "python",
                signal: controller.signal,
              }),
        )
        .then((raw) => {
          if (
            !this.alive ||
            generation !== this.generation ||
            controller.signal.aborted ||
            this.view.state.doc !== state.doc ||
            this.view.state.selection.main.head !== pos ||
            !sameConfiguration(config, this.view.state.facet(configuration)) ||
            !this.eligible()
          )
            return;
          const text = normalizeInlineSuggestion(raw, prefix, suffix);
          if (text)
            this.view.dispatch({ effects: setSuggestion.of({ pos, text }) });
        })
        .catch(() => {
          /* An unavailable provider leaves ordinary editing untouched. */
        })
        .finally(() => {
          if (generation === this.generation) {
            if (this.timeoutTimer) clearTimeout(this.timeoutTimer);
            this.timeoutTimer = null;
            this.controller = null;
          }
        });
    }

    private cancel() {
      this.generation++;
      if (this.idleTimer) clearTimeout(this.idleTimer);
      if (this.timeoutTimer) clearTimeout(this.timeoutTimer);
      this.idleTimer = this.timeoutTimer = null;
      this.controller?.abort();
      this.controller = null;
    }
    isVisible() {
      return this.decorations.size > 0;
    }
    destroy() {
      this.alive = false;
      this.cancel();
    }
  },
  { decorations: (value) => value.decorations },
);

export function cancelInlineSuggestion(view: EditorView) {
  if (view.state.field(suggestionState, false) !== undefined)
    view.dispatch({ effects: dismissSuggestion.of(null) });
}

export function inlineSuggestionExtensions(
  options: InlineSuggestionOptions = {},
): Extension {
  return [
    configuration.of(options),
    suggestionState,
    inlineController,
    Prec.highest(
      keymap.of([
        {
          key: "Tab",
          run: (view) => {
            const suggestion = view.state.field(suggestionState);
            if (
              !suggestion ||
              view.state.readOnly ||
              completionStatus(view.state) !== null ||
              !view.hasFocus ||
              !view.plugin(inlineController)?.isVisible()
            )
              return false;
            view.dispatch({
              changes: { from: suggestion.pos, insert: suggestion.text },
              selection: { anchor: suggestion.pos + suggestion.text.length },
              userEvent: "input.complete",
            });
            return true;
          },
        },
        {
          key: "Escape",
          run: (view) => {
            if (completionStatus(view.state) !== null) return false;
            const visible = view.plugin(inlineController)?.isVisible() ?? false;
            cancelInlineSuggestion(view);
            return visible;
          },
        },
      ]),
    ),
    EditorView.baseTheme({
      ".cm-inline-suggestion": {
        color: "#718198",
        opacity: "0.85",
        whiteSpace: "pre",
        pointerEvents: "none",
        userSelect: "none",
      },
    }),
  ];
}
