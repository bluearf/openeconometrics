import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { basicSetup } from "codemirror";
import { Compartment, EditorState } from "@codemirror/state";
import { closeCompletion } from "@codemirror/autocomplete";
import { python } from "@codemirror/lang-python";
import { markdown } from "@codemirror/lang-markdown";
import { stex } from "@codemirror/legacy-modes/mode/stex";
import {
  HighlightStyle,
  StreamLanguage,
  syntaxHighlighting,
} from "@codemirror/language";
import { EditorView, keymap, tooltips } from "@codemirror/view";
import { indentWithTab } from "@codemirror/commands";
import { tags } from "@lezer/highlight";
import { dismissEditorHelp, editorHelpExtensions } from "./editor-help";
import * as inline from "./editor-inline";
import { qaBeforeInput, qaTyped, qaPaint } from "./qa-performance";

const editorHighlighting = HighlightStyle.define([
  { tag: tags.keyword, color: "#142d50", fontWeight: "600" },
  { tag: [tags.name, tags.operator, tags.punctuation], color: "#203b5d" },
  { tag: tags.literal, color: "#345a82" },
  { tag: tags.comment, color: "#63748b", fontStyle: "italic" },
  { tag: tags.meta, color: "#4c6380" },
  { tag: tags.invalid, textDecoration: "underline wavy #142d50" },
]);

export interface EditorHandle {
  getCode: () => string;
  getSelectionOrLine: () => string;
  insert: (text: string) => void;
  focus: () => void;
  cancelInlineSuggestion: () => void;
}
interface Props {
  /** Stable project and file identity, independent of completion settings. */
  documentKey?: string;
  language?: "python" | "markdown" | "latex";
  value: string;
  onChange: (value: string) => void;
  onRun: (code: string) => void;
  onCursor: (line: number, column: number) => void;
  readOnly?: boolean;
  fullScriptOnly?: boolean;
  inlineSuggestionProvider?: inline.InlineSuggestionProvider;
  inlineSuggestionContextKey?: string;
}

const CodeEditor = forwardRef<EditorHandle, Props>(
  function CodeEditor(props, ref) {
    const host = useRef<HTMLDivElement>(null);
    const view = useRef<EditorView | null>(null);
    const access = useRef(new Compartment());
    const suggestions = useRef(new Compartment());
    const languageMode = useRef(new Compartment());
    const helpContext = useRef(props.inlineSuggestionContextKey);
    const callbacks = useRef(props);
    callbacks.current = props;
    const activeDocument = useRef(props.documentKey);
    const createState = useRef<((doc: string) => EditorState) | null>(null);
    const documents = useRef(
      new Map<
        string | undefined,
        { state: EditorState; scroll: ReturnType<EditorView["scrollSnapshot"]> }
      >(),
    );
    const selected = () => {
      const state = view.current?.state;
      if (!state) return "";
      const { from, to } = state.selection.main;
      return from === to
        ? state.doc.lineAt(from).text
        : state.sliceDoc(from, to);
    };
    const isPython = () =>
      !callbacks.current.language || callbacks.current.language === "python";
    const languageExtensions = (language = "python") => [
      language === "markdown"
        ? markdown()
        : language === "latex"
          ? StreamLanguage.define(stex)
          : python(),
      ...(language === "python" ? editorHelpExtensions() : []),
      EditorView.contentAttributes.of({
        "aria-label": "File editor",
        ...(language === "python"
          ? {
              "aria-description":
                "Function help: F1. Autocomplete: Control and Space. Inline suggestions: Tab to accept, Escape to dismiss.",
            }
          : {}),
        spellcheck: "false",
      }),
    ];
    useImperativeHandle(ref, () => ({
      getCode: () => view.current?.state.doc.toString() || "",
      getSelectionOrLine: selected,
      insert: (text: string) => {
        const editor = view.current;
        if (!editor || callbacks.current.readOnly) return;
        const end = editor.state.doc.length;
        editor.dispatch({
          changes: { from: end, insert: `${end ? "\n" : ""}${text}\n` },
          selection: { anchor: end + (end ? 1 : 0) + text.length },
        });
        editor.focus();
      },
      focus: () => view.current?.focus(),
      cancelInlineSuggestion: () => {
        if (view.current) {
          inline.cancelInlineSuggestion(view.current);
          dismissEditorHelp(view.current);
        }
      },
    }));
    useEffect(() => {
      if (!host.current) return;
      const extensions = [
        access.current.of([
          EditorState.readOnly.of(Boolean(callbacks.current.readOnly)),
          EditorView.editable.of(!callbacks.current.readOnly),
        ]),
        keymap.of([
          {
            key: "Mod-Enter",
            run: (v) => {
              inline.cancelInlineSuggestion(v);
              dismissEditorHelp(v);
              if (!callbacks.current.readOnly && isPython())
                callbacks.current.onRun(
                  callbacks.current.fullScriptOnly
                    ? view.current?.state.doc.toString() || ""
                    : selected(),
                );
              return true;
            },
          },
          {
            key: "Mod-Shift-Enter",
            run: (v) => {
              inline.cancelInlineSuggestion(v);
              dismissEditorHelp(v);
              if (!callbacks.current.readOnly && isPython())
                callbacks.current.onRun(v.state.doc.toString());
              return true;
            },
          },
          indentWithTab,
        ]),
        basicSetup,
        languageMode.current.of(languageExtensions(callbacks.current.language)),
        // WebKit can position tooltips absolutely; keep them outside the
        // resizable panels' clipping and scrolling containers.
        tooltips({ parent: host.current.ownerDocument.body }),
        suggestions.current.of(
          inline.inlineSuggestionExtensions({
            provider: isPython()
              ? callbacks.current.inlineSuggestionProvider
              : undefined,
            contextKey: callbacks.current.inlineSuggestionContextKey,
          }),
        ),
        syntaxHighlighting(editorHighlighting),
        EditorView.domEventHandlers({
          beforeinput(event) {
            if (event.isTrusted) qaBeforeInput();
            return false;
          },
        }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) qaTyped();
          if (update.docChanged)
            callbacks.current.onChange(update.state.doc.toString());
          if (update.selectionSet || update.docChanged) {
            const pos = update.state.selection.main.head;
            const line = update.state.doc.lineAt(pos);
            callbacks.current.onCursor(line.number, pos - line.from + 1);
          }
        }),
        EditorView.theme({
          "&": {
            height: "100%",
            fontSize: "13px",
            backgroundColor: "#fff",
            color: "#172d4c",
          },
          ".cm-scroller": {
            overflow: "auto",
            fontFamily:
              '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
            lineHeight: "1.7",
          },
          ".cm-content": { padding: "14px 0", caretColor: "#172d4c" },
          ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#172d4c" },
          ".cm-line": { paddingLeft: "15px", paddingRight: "20px" },
          ".cm-gutters": {
            background: "#fff",
            color: "#63748b",
            border: "none",
            paddingLeft: "10px",
            paddingRight: "7px",
          },
          ".cm-activeLine": { background: "#f6f8fb" },
          ".cm-activeLineGutter": { background: "#f6f8fb", color: "#172d4c" },
          "&.cm-focused": {
            outline: "1px solid #315478",
            outlineOffset: "-1px",
          },
          "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
            background: "#dce6f3 !important",
          },
          ".cm-selectionMatch, .cm-searchMatch": { background: "#e7edf6" },
          ".cm-searchMatch.cm-searchMatch-selected": {
            background: "#cbd9ed",
          },
          ".cm-matchingBracket": { background: "#dce6f3", color: "#142d50" },
          ".cm-panels": { background: "#fff", color: "#172d4c" },
          ".cm-panels-top": { borderBottom: "1px solid #d6dfeb" },
          ".cm-panels-bottom": { borderTop: "1px solid #d6dfeb" },
          ".cm-tooltip": {
            background: "#fff",
            color: "#172d4c",
            border: "1px solid #d6dfeb",
            boxShadow: "0 4px 18px #172d4c14",
          },
          ".cm-tooltip-autocomplete > ul > li[aria-selected]": {
            background: "#e7edf6",
            color: "#142d50",
          },
          ".cm-button": {
            backgroundImage: "none",
            backgroundColor: "#fff",
            color: "#172d4c",
            border: "1px solid #cbd7e6",
            borderRadius: "4px",
          },
          ".cm-button:hover": { backgroundColor: "#f0f4f9" },
          ".cm-textfield": {
            background: "#fff",
            color: "#172d4c",
            border: "1px solid #cbd7e6",
          },
          ".cm-button:focus-visible, .cm-textfield:focus-visible": {
            outline: "2px solid #315478",
            outlineOffset: "1px",
          },
          ".cm-foldGutter": { width: "10px" },
        }),
      ];
      createState.current = (doc) => EditorState.create({ doc, extensions });
      const editor = new EditorView({
        state: createState.current(callbacks.current.value),
        parent: host.current,
      });
      view.current = editor;
      return () => {
        editor.destroy();
        view.current = null;
        createState.current = null;
        documents.current.clear();
      };
    }, []);
    useEffect(() => {
      const editor = view.current;
      if (!editor) return;
      if (activeDocument.current !== props.documentKey) {
        inline.cancelInlineSuggestion(editor);
        dismissEditorHelp(editor);
        closeCompletion(editor);
        // Cache editor state, not a remote draft: incoming source is always
        // validated by the workspace before a saved state can be reused.
        documents.current.delete(activeDocument.current);
        documents.current.set(activeDocument.current, {
          state: editor.state,
          scroll: editor.scrollSnapshot(),
        });
        // Bound retained histories even when a project has many files.
        while (documents.current.size > 32)
          documents.current.delete(documents.current.keys().next().value);
        const cached = documents.current.get(props.documentKey);
        const reuse = cached?.state.doc.toString() === props.value;
        const next = reuse ? cached!.state : createState.current!(props.value);
        activeDocument.current = props.documentKey;
        qaPaint("file");
        editor.setState(
          next.update({
            effects: [
              access.current.reconfigure([
                EditorState.readOnly.of(Boolean(props.readOnly)),
                EditorView.editable.of(!props.readOnly),
              ]),
              suggestions.current.reconfigure(
                inline.inlineSuggestionExtensions({
                  provider: isPython()
                    ? props.inlineSuggestionProvider
                    : undefined,
                  contextKey: props.inlineSuggestionContextKey,
                }),
              ),
              languageMode.current.reconfigure(
                languageExtensions(props.language),
              ),
            ],
          }).state,
        );
        if (reuse) editor.dispatch({ effects: cached!.scroll });
        else editor.scrollDOM.scrollTop = editor.scrollDOM.scrollLeft = 0;
        const pos = editor.state.selection.main.head;
        const line = editor.state.doc.lineAt(pos);
        callbacks.current.onCursor(line.number, pos - line.from + 1);
        return;
      }
      if (editor.state.doc.toString() !== props.value) {
        editor.dispatch({
          changes: {
            from: 0,
            to: editor.state.doc.length,
            insert: props.value,
          },
        });
        dismissEditorHelp(editor);
      }
    }, [props.value, props.documentKey]);
    useEffect(() => {
      if (props.readOnly && view.current) dismissEditorHelp(view.current);
      view.current?.dispatch({
        effects: access.current.reconfigure([
          EditorState.readOnly.of(Boolean(props.readOnly)),
          EditorView.editable.of(!props.readOnly),
        ]),
      });
    }, [props.readOnly]);
    useEffect(() => {
      if (helpContext.current !== props.inlineSuggestionContextKey) {
        helpContext.current = props.inlineSuggestionContextKey;
        if (view.current) dismissEditorHelp(view.current);
      }
    }, [props.inlineSuggestionContextKey]);
    useEffect(() => {
      if (view.current) {
        inline.cancelInlineSuggestion(view.current);
        dismissEditorHelp(view.current);
        closeCompletion(view.current);
      }
      view.current?.dispatch({
        effects: languageMode.current.reconfigure(
          languageExtensions(props.language),
        ),
      });
    }, [props.language]);
    useEffect(() => {
      view.current?.dispatch({
        effects: suggestions.current.reconfigure(
          inline.inlineSuggestionExtensions({
            provider: isPython() ? props.inlineSuggestionProvider : undefined,
            contextKey: props.inlineSuggestionContextKey,
          }),
        ),
      });
    }, [
      props.inlineSuggestionProvider,
      props.inlineSuggestionContextKey,
      props.language,
    ]);
    return <div className="code-editor" ref={host} />;
  },
);
export default CodeEditor;
