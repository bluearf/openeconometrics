import {
  StateEffect,
  StateField,
  type EditorState,
  type Extension,
} from "@codemirror/state";
import { ensureSyntaxTree, syntaxTree } from "@codemirror/language";
import { globalCompletion, pythonLanguage } from "@codemirror/lang-python";
import {
  activateHover,
  closeHoverTooltip,
  closeHoverTooltips,
  EditorView,
  hoverTooltip,
  keymap,
  showTooltip,
  ViewPlugin,
  type ViewUpdate,
  type Tooltip,
} from "@codemirror/view";
import {
  autocompletion,
  type Completion,
  type CompletionContext,
} from "@codemirror/autocomplete";
import { API_CATALOG, type ApiEntry, type ApiParameter } from "./editor-api.ts";

type SyntaxNode = ReturnType<typeof syntaxTree>["topNode"];
interface Binding {
  name: string;
  from: number;
  canonical?: string;
  entry?: ApiEntry;
  instance?: boolean;
}
interface Scope {
  from: number;
  to: number;
  parent?: Scope;
  bindings: Binding[];
  names: Map<string, Binding[]>;
  children: Scope[];
  declared?: Set<string>;
}
interface SemanticIndex {
  tree: ReturnType<typeof syntaxTree>;
  code: string;
  root: Scope;
}
const indexes = new WeakMap<EditorState["doc"], SemanticIndex>();

function children(node: SyntaxNode): SyntaxNode[] {
  const nodes: SyntaxNode[] = [];
  for (let child = node.firstChild; child; child = child.nextSibling)
    nodes.push(child);
  return nodes;
}
function text(index: SemanticIndex, node: SyntaxNode): string {
  return index.code.slice(node.from, node.to);
}
function blocked(node: SyntaxNode | null): boolean {
  for (; node; node = node.parent)
    if (/String|Comment/.test(node.name)) return true;
  return false;
}
function scopeAt(root: Scope, pos: number): Scope {
  const child = root.children.find(
    (scope) => pos >= scope.from && pos <= scope.to,
  );
  return child ? scopeAt(child, pos) : root;
}
function addBinding(scope: Scope, binding: Binding) {
  scope.bindings.push(binding);
  const named = scope.names.get(binding.name);
  if (named) named.push(binding);
  else scope.names.set(binding.name, [binding]);
}
function bindingAt(
  scope: Scope,
  name: string,
  pos: number,
): Binding | undefined {
  const named = scope.names.get(name);
  if (named) {
    for (let at = named.length - 1; at >= 0; at--)
      if (named[at].from <= pos) return named[at];
    // Python treats an assigned name as local throughout a function body.
    if (scope.parent) return { name, from: scope.from };
  }
  if (scope.parent && scope.declared?.has(name))
    return { name, from: scope.from };
  return scope.parent ? bindingAt(scope.parent, name, pos) : undefined;
}
function canonicalImport(name: string): string {
  return name
    .replace(/^openecon\.plotting(?=\.|$)/, "openecon.plot")
    .replace(/^openecon\.frame\.DataFrame$/, "openecon.DataFrame")
    .replace(/^openecon\.models\.ResultBundle$/, "openecon.ResultBundle");
}
function resolveName(
  index: SemanticIndex,
  scope: Scope,
  expression: string,
  pos: number,
): Binding | undefined {
  const parts = expression.replace(/\s+/g, "").split(".");
  if (!parts.every((part) => /^[\p{ID_Start}_][\p{ID_Continue}]*$/u.test(part)))
    return;
  const binding = bindingAt(scope, parts[0], pos);
  if (binding) {
    if (parts.length === 1) return binding;
    if (!binding.canonical) return;
    const canonical = canonicalImport(
      [binding.canonical, ...parts.slice(1)].join("."),
    );
    return {
      name: expression,
      from: pos,
      canonical,
      entry: API_CATALOG[canonical],
    };
  }
  const globalName =
    parts[0] === "display" ? "console.display" : `builtins.${parts[0]}`;
  if (parts.length === 1 && API_CATALOG[globalName]) {
    const canonical = globalName;
    return {
      name: parts[0],
      from: pos,
      canonical,
      entry: API_CATALOG[canonical],
    };
  }
  // An unimported module or arbitrary object must never receive invented API help.
  void index;
}
function resolveNode(
  index: SemanticIndex,
  scope: Scope,
  node: SyntaxNode,
  pos: number,
): Binding | undefined {
  if (node.name === "VariableName" || node.name === "MemberExpression") {
    return resolveName(index, scope, text(index, node), pos);
  }
  if (node.name === "CallExpression" && node.firstChild) {
    const callable = resolveNode(index, scope, node.firstChild, pos);
    const returns = callable?.entry?.returns;
    if (returns && /^openecon\.[A-Za-z_.]+$/.test(returns)) {
      return {
        name: "",
        from: pos,
        canonical: returns,
        entry: API_CATALOG[returns],
        instance: true,
      };
    }
  }
}
function listChildren(list: SyntaxNode): SyntaxNode[] {
  return children(list).flatMap((node) =>
    node.name === "⚠"
      ? children(node).filter((part) => part.name === ",")
      : [node],
  );
}
function parameterSegments(list: SyntaxNode): SyntaxNode[][] {
  const segments: SyntaxNode[][] = [[]];
  for (const node of listChildren(list)) {
    if (node.name === ",") segments.push([]);
    else if (!["(", ")", "⚠", "Comment"].includes(node.name))
      segments.at(-1)!.push(node);
  }
  return segments;
}
function userFunction(
  index: SemanticIndex,
  node: SyntaxNode,
): ApiEntry | undefined {
  const nodes = children(node),
    name = nodes.find((child) => child.name === "VariableName");
  const params = nodes.find((child) => child.name === "ParamList");
  if (!name || !params) return;
  const parameters: ApiEntry["parameters"] = [];
  let keywordOnly = false;
  for (const segment of parameterSegments(params)) {
    if (segment.some((part) => part.name === "/")) {
      for (const parameter of parameters) parameter.kind = "positional-only";
      continue;
    }
    const variable = segment.find((part) => part.name === "VariableName");
    if (!variable) {
      if (segment.some((part) => part.name === "*")) keywordOnly = true;
      continue;
    }
    const prefix = segment[0].name;
    const assigned = segment.findIndex((part) => part.name === "AssignOp");
    parameters.push({
      name: text(index, variable),
      ...(assigned >= 0
        ? {
            default: segment
              .slice(assigned + 1)
              .map((part) => text(index, part))
              .join(""),
          }
        : {}),
      kind:
        prefix === "**"
          ? "var-keyword"
          : prefix === "*"
            ? "var-positional"
            : keywordOnly
              ? "keyword-only"
              : "positional-or-keyword",
    });
    if (prefix === "*") keywordOnly = true;
  }
  const body = nodes.find((child) => child.name === "Body");
  const firstStatement =
    body &&
    children(body).find(
      (child) => child.name !== ":" && child.name !== "Comment",
    );
  const string =
    firstStatement?.name === "ExpressionStatement" &&
    firstStatement.firstChild?.name === "String"
      ? firstStatement.firstChild
      : undefined;
  let description = "";
  if (string) {
    const raw = text(index, string),
      quote = raw.match(/^[rRuU]*("""|'''|"|')/)?.[1];
    if (quote) {
      description = raw
        .slice(raw.indexOf(quote) + quote.length, -quote.length)
        .replace(/\\n/g, "\n")
        .split(/\n\s*\n/)[0]
        .replace(/\s+/g, " ")
        .trim();
      if (description.length > 400)
        description = `${description.slice(0, 397)}…`;
    }
  }
  return {
    name: text(index, name),
    signature: `${text(index, name)}${text(index, params).replace(/\s+/g, " ")}`,
    description,
    parameters,
    kind: "function",
  };
}
function collectImport(index: SemanticIndex, scope: Scope, node: SyntaxNode) {
  const nodes = children(node),
    fromIndex = nodes.findIndex((part) => part.name === "from");
  const importIndex = nodes.findIndex((part) => part.name === "import");
  if (importIndex < 0) return;
  const module =
    fromIndex >= 0
      ? nodes
          .slice(fromIndex + 1, importIndex)
          .map((part) => text(index, part))
          .join("")
      : "";
  const segments: SyntaxNode[][] = [[]];
  for (const part of nodes.slice(importIndex + 1)) {
    if (part.name === ",") segments.push([]);
    else if (!["(", ")", "Comment", "⚠"].includes(part.name))
      segments.at(-1)!.push(part);
  }
  for (const segment of segments) {
    const asIndex = segment.findIndex((part) => part.name === "as");
    const imported = segment
      .slice(0, asIndex < 0 ? undefined : asIndex)
      .map((part) => text(index, part))
      .join("");
    const alias =
      asIndex < 0
        ? module
          ? imported
          : imported.split(".")[0]
        : segment[asIndex + 1] && text(index, segment[asIndex + 1]);
    if (!alias || !/^[\p{ID_Start}_][\p{ID_Continue}]*$/u.test(alias)) continue;
    const canonical = canonicalImport(
      module
        ? `${module}.${imported}`
        : asIndex < 0
          ? imported.split(".")[0]
          : imported,
    );
    addBinding(scope, {
      name: alias,
      from: node.from,
      canonical,
      entry: API_CATALOG[canonical],
    });
  }
}
function targetVariables(nodes: SyntaxNode[]): SyntaxNode[] {
  return nodes.flatMap((node) =>
    node.name === "VariableName"
      ? [node]
      : /^(TupleExpression|ArrayExpression|StarredExpression)$/.test(node.name)
        ? targetVariables(children(node))
        : [],
  );
}
function declaredNames(index: SemanticIndex, body: SyntaxNode): Set<string> {
  const names = new Set<string>(),
    external = new Set<string>();
  const add = (node: SyntaxNode) => names.add(text(index, node));
  const visit = (node: SyntaxNode) => {
    if (node.name === "FunctionDefinition" || node.name === "ClassDefinition") {
      const name = children(node).find((part) => part.name === "VariableName");
      if (name) add(name);
      return;
    }
    if (
      node.name === "LambdaExpression" ||
      /ComprehensionExpression$/.test(node.name) ||
      blocked(node)
    )
      return;
    const nodes = children(node);
    if (node.name === "ImportStatement") {
      const temporary: Scope = {
        from: node.from,
        to: node.to,
        bindings: [],
        names: new Map(),
        children: [],
      };
      collectImport(index, temporary, node);
      for (const binding of temporary.bindings) names.add(binding.name);
      return;
    }
    if (node.name === "ScopeStatement") {
      for (const part of nodes)
        if (part.name === "VariableName") external.add(text(index, part));
      return;
    }
    if (node.name === "AssignStatement") {
      let from = 0;
      for (let at = 0; at < nodes.length; at++) {
        if (nodes[at].name !== "AssignOp") continue;
        targetVariables(nodes.slice(from, at)).forEach(add);
        from = at + 1;
      }
      if (!from) targetVariables(nodes).forEach(add);
    } else if (
      ["UpdateStatement", "DeleteStatement", "NamedExpression"].includes(
        node.name,
      )
    ) {
      const target = nodes.find((part) => part.name === "VariableName");
      if (target) add(target);
    } else if (node.name === "ForStatement") {
      const end = nodes.findIndex((part) => part.name === "in");
      targetVariables(nodes.slice(0, end < 0 ? 0 : end)).forEach(add);
    } else if (node.name === "WithStatement" || node.name === "TryStatement") {
      for (let at = 0; at < nodes.length - 1; at++)
        if (nodes[at].name === "as" && nodes[at + 1].name === "VariableName")
          add(nodes[at + 1]);
    } else if (
      node.name === "PrintStatement" &&
      /^print\s*(?:=|[+*/%&|^-]=)/.test(text(index, node))
    )
      names.add("print");
    for (const child of nodes) visit(child);
  };
  visit(body);
  for (const name of external) names.delete(name);
  return names;
}
function collect(index: SemanticIndex, scope: Scope, node: SyntaxNode): void {
  if (node.name === "ImportStatement") {
    collectImport(index, scope, node);
    return;
  }
  if (node.name === "FunctionDefinition") {
    const entry = userFunction(index, node);
    if (entry) addBinding(scope, { name: entry.name, from: node.from, entry });
    const body = children(node).find((child) => child.name === "Body");
    if (body) {
      const local: Scope = {
        from: body.from,
        to: body.to,
        parent: scope,
        bindings: [],
        names: new Map(),
        children: [],
        declared: declaredNames(index, body),
      };
      for (const parameter of entry?.parameters ?? [])
        addBinding(local, { name: parameter.name, from: body.from });
      scope.children.push(local);
      for (const child of children(body)) collect(index, local, child);
    }
    return;
  }
  if (node.name === "ClassDefinition") {
    const name = children(node).find((child) => child.name === "VariableName");
    if (name) addBinding(scope, { name: text(index, name), from: node.from });
    return;
  }
  if (node.name === "LambdaExpression") {
    const nodes = children(node),
      colon = nodes.findIndex((child) => child.name === ":");
    const params = nodes.find((child) => child.name === "ParamList");
    if (colon >= 0) {
      const local: Scope = {
        from: nodes[colon].to,
        to: node.to,
        parent: scope,
        bindings: [],
        names: new Map(),
        children: [],
      };
      scope.children.push(local);
      if (params) {
        for (const segment of parameterSegments(params)) {
          const variable = segment.find(
            (child) => child.name === "VariableName",
          );
          if (variable)
            addBinding(local, {
              name: text(index, variable),
              from: local.from,
            });
        }
      }
      for (const child of nodes.slice(colon + 1)) collect(index, local, child);
    }
    return;
  }
  if (/ComprehensionExpression$/.test(node.name)) {
    const local: Scope = {
      from: node.from,
      to: node.to,
      parent: scope,
      bindings: [],
      names: new Map(),
      children: [],
    };
    scope.children.push(local);
    let target = false;
    for (const child of children(node)) {
      if (child.name === "for") target = true;
      else if (child.name === "in") target = false;
      else if (target && child.name === "VariableName")
        addBinding(local, { name: text(index, child), from: node.from });
    }
    for (const child of children(node))
      if (!blocked(child)) collect(index, local, child);
    return;
  }
  if (
    node.name === "PrintStatement" &&
    /^print\s*(?:=|[+*/%&|^-]=)/.test(text(index, node))
  ) {
    // Lezer also recognizes Python 2's print statement; Python 3 permits rebinding it.
    addBinding(scope, { name: "print", from: node.to });
    return;
  }
  if (node.name === "AssignStatement") {
    const nodes = children(node),
      operator = nodes.map((child) => child.name).lastIndexOf("AssignOp");
    if (operator > 0) {
      const value = nodes[operator + 1];
      const inferred = value
        ? resolveNode(index, scope, value, node.from)
        : undefined;
      // Chained simple targets receive the result type; unpacked values are unknown.
      let start = 0;
      for (let end = 0; end <= operator; end++) {
        if (nodes[end].name !== "AssignOp") continue;
        const targets = nodes.slice(start, end);
        const simple =
          targets[0]?.name === "VariableName" &&
          targets.every(
            (target) =>
              target.name === "VariableName" || target.name === "TypeDef",
          );
        for (const target of targetVariables(targets)) {
          addBinding(scope, {
            name: text(index, target),
            from: node.to,
            canonical: simple ? inferred?.canonical : undefined,
            entry: simple ? inferred?.entry : undefined,
            instance: simple ? inferred?.instance : undefined,
          });
        }
        start = end + 1;
      }
      if (value) collect(index, scope, value);
    }
    return;
  }
  if (
    node.name === "UpdateStatement" ||
    node.name === "DeleteStatement" ||
    node.name === "NamedExpression"
  ) {
    const target = children(node).find(
      (child) => child.name === "VariableName",
    );
    if (target) addBinding(scope, { name: text(index, target), from: node.to });
    return;
  }
  if (node.name === "WithStatement" || node.name === "TryStatement") {
    const nodes = children(node);
    for (let at = 0; at < nodes.length - 1; at++) {
      if (nodes[at].name === "as" && nodes[at + 1].name === "VariableName") {
        addBinding(scope, {
          name: text(index, nodes[at + 1]),
          from: nodes[at + 1].to,
        });
      }
    }
  }
  if (node.name === "ForStatement") {
    for (const child of children(node)) {
      if (child.name === "in") break;
      if (child.name === "VariableName")
        addBinding(scope, { name: text(index, child), from: child.to });
    }
  }
  for (const child of children(node))
    if (!blocked(child)) collect(index, scope, child);
}
function semanticIndex(state: EditorState, pos: number): SemanticIndex {
  let tree = syntaxTree(state);
  const needed = Math.min(state.doc.length, Math.max(0, pos + 1));
  if (tree.length < needed) {
    // A cold or busy editor may not yet have parsed the requested token.
    // Advance only as far as that token, within a bounded interactive budget.
    tree = ensureSyntaxTree(state, needed, 20) ?? tree;
  }
  const cached = indexes.get(state.doc);
  if (cached?.tree === tree) return cached;
  const index: SemanticIndex = {
    tree,
    code: state.doc.toString(),
    root: {
      from: 0,
      to: state.doc.length,
      bindings: [],
      names: new Map(),
      children: [],
    },
  };
  collect(index, index.root, tree.topNode);
  indexes.set(state.doc, index);
  return index;
}

export interface EditorSymbol {
  entry: ApiEntry;
  from: number;
  to: number;
  expression: string;
}
export function getEditorSymbol(
  state: EditorState,
  pos: number,
  side: -1 | 1 = 1,
): EditorSymbol | null {
  const index = semanticIndex(state, pos);
  let node = index.tree.resolveInner(pos, side);
  if (blocked(node)) return null;
  if (!["VariableName", "PropertyName"].includes(node.name)) return null;
  const token = node;
  if (node.name === "PropertyName" && node.parent?.name === "MemberExpression")
    node = node.parent;
  const binding = resolveNode(index, scopeAt(index.root, pos), node, pos);
  return binding?.entry
    ? {
        entry: binding.entry,
        from: token.from,
        to: token.to,
        expression: text(index, node),
      }
    : null;
}

export interface ActiveCall {
  entry: ApiEntry;
  from: number;
  to: number;
  argumentIndex: number;
  activeParameter?: string;
  supplied: string[];
  argumentFrom: number;
}
export function getActiveCall(
  state: EditorState,
  pos = state.selection.main.head,
): ActiveCall | null {
  const index = semanticIndex(state, pos);
  let node: SyntaxNode | null = index.tree.resolveInner(pos, -1);
  if (node.name === "Comment") return null;
  for (; node; node = node.parent) {
    if (
      node.name !== "ArgList" ||
      node.parent?.name !== "CallExpression" ||
      pos <= node.from
    )
      continue;
    const nodes = listChildren(node),
      closing = nodes.find((part) => part.name === ")");
    if (closing && pos > closing.from) continue;
    const callable = node.parent.firstChild;
    if (!callable) continue;
    const binding = resolveNode(index, scopeAt(index.root, pos), callable, pos);
    if (!binding?.entry) continue;
    const segments = parameterSegments(node),
      commas = nodes.filter((part) => part.name === "," && part.to <= pos);
    const argumentIndex = commas.length,
      current = segments[argumentIndex] ?? [];
    const keyword = (parts: SyntaxNode[]) =>
      parts[0]?.name === "VariableName" && parts[1]?.name === "AssignOp"
        ? text(index, parts[0])
        : undefined;
    const activeKeyword = keyword(current);
    const positionalIndex = segments
      .slice(0, argumentIndex)
      .filter((parts) => !keyword(parts)).length;
    const positional = binding.entry.parameters.filter(
      (parameter) =>
        parameter.kind !== "keyword-only" && parameter.kind !== "var-keyword",
    );
    let priorPosition = 0;
    const supplied = segments.flatMap((parts, at) => {
      if (at === argumentIndex) return [];
      const name = keyword(parts);
      if (name) return [name];
      if (at > argumentIndex) return [];
      const parameter = positional[priorPosition++];
      return parameter && parameter.kind !== "var-positional"
        ? [parameter.name]
        : [];
    });
    const candidate = activeKeyword
      ? (binding.entry.parameters.find(
          (parameter) => parameter.name === activeKeyword,
        ) ??
        binding.entry.parameters.find(
          (parameter) => parameter.kind === "var-keyword",
        ))
      : (positional[positionalIndex] ??
        positional.find((parameter) => parameter.kind === "var-positional"));
    const activeParameter =
      candidate && !supplied.includes(candidate.name)
        ? candidate.name
        : undefined;
    return {
      entry: binding.entry,
      from: callable.from,
      to: callable.to,
      argumentIndex,
      activeParameter,
      supplied,
      argumentFrom: commas.at(-1)?.to ?? node.from + 1,
    };
  }
  return null;
}

const signatureRanges = new WeakMap<
  ApiEntry,
  Map<string, { from: number; to: number }>
>();
function parameterRange(entry: ApiEntry, name: string) {
  let ranges = signatureRanges.get(entry);
  if (!ranges) {
    ranges = new Map();
    const at = entry.signature.indexOf("("),
      prefix = "def _help";
    if (at >= 0) {
      const functionNode = pythonLanguage.parser.parse(
        `${prefix}${entry.signature.slice(at)}:\n    pass`,
      ).topNode.firstChild;
      const params =
        functionNode &&
        children(functionNode).find((node) => node.name === "ParamList");
      if (params) {
        for (const segment of parameterSegments(params)) {
          const variable = segment.find((node) => node.name === "VariableName");
          if (!variable) continue;
          const from = variable.from + at - prefix.length,
            to = variable.to + at - prefix.length;
          ranges.set(entry.signature.slice(from, to), { from, to });
        }
      }
    }
    signatureRanges.set(entry, ranges);
  }
  return ranges.get(name);
}
function helpDOM(entry: ApiEntry, activeParameter?: string): HTMLElement {
  const dom = document.createElement("div");
  dom.className = "cm-editor-help";
  dom.setAttribute("role", "tooltip");
  dom.setAttribute(
    "aria-label",
    `${entry.signature}${activeParameter ? ` · ${activeParameter}` : ""}`,
  );
  const signature = document.createElement("div");
  signature.className = "cm-help-signature";
  const range = activeParameter
    ? parameterRange(entry, activeParameter)
    : undefined;
  if (range) {
    signature.append(
      document.createTextNode(entry.signature.slice(0, range.from)),
    );
    const strong = document.createElement("strong");
    strong.className = "cm-help-active-parameter";
    strong.textContent = activeParameter!;
    signature.append(
      strong,
      document.createTextNode(entry.signature.slice(range.to)),
    );
  } else signature.textContent = entry.signature;
  dom.append(signature);
  if (entry.description) {
    const description = document.createElement("p");
    description.textContent = entry.description;
    dom.append(description);
  }
  const detail = entry.parameters.find(
    (parameter) => parameter.name === activeParameter,
  )?.description;
  if (detail) {
    const parameter = document.createElement("p");
    parameter.className = "cm-help-parameter";
    parameter.textContent = `${activeParameter}: ${detail}`;
    dom.append(parameter);
  }
  return dom;
}
function asCompletion(entry: ApiEntry, label: string): Completion {
  return {
    label,
    type: entry.kind,
    detail: entry.signature.slice(entry.signature.indexOf("(")),
    info: () => helpDOM(entry),
    boost: 2,
  };
}

// Only inert scalar literals belong in value suggestions. Never evaluate a
// user's default expression, import packages, or derive values from prose.
const scalarLiteral =
  /^(?:None|True|False|[-+]?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|'(?:[^'\\\r\n]|\\.)*'|"(?:[^"\\\r\n]|\\.)*")$/;
function parameterLiterals(parameter: ApiParameter | undefined): string[] {
  if (!parameter) return [];
  const choices = [...(parameter.choices ?? [])];
  if (parameter.default && scalarLiteral.test(parameter.default)) {
    if (parameter.default === "True" || parameter.default === "False")
      choices.push("True", "False");
    else choices.push(parameter.default);
  }
  return [...new Set(choices)].filter((choice) => scalarLiteral.test(choice));
}
function plainString(node: SyntaxNode | null): SyntaxNode | undefined {
  for (; node; node = node.parent) {
    if (node.name === "String") return node;
    if (/String|Comment/.test(node.name)) return;
  }
}
function stringContent(literal: string, quote: string): string | undefined {
  if (!scalarLiteral.test(literal) || !/^['"]/.test(literal)) return;
  const content = literal.slice(1, -1);
  let result = "";
  for (let at = 0; at < content.length; at++) {
    const char = content[at];
    if (char === "\\") {
      result += char + content[++at];
    } else result += char === quote ? `\\${char}` : char;
  }
  return result;
}
function stringCompletions(
  context: CompletionContext,
  index: SemanticIndex,
  node: SyntaxNode,
  call: ActiveCall,
) {
  if (node.parent?.name !== "ArgList") return null;
  const before = context.state.sliceDoc(call.argumentFrom, node.from);
  if (!/^\s*(?:[\p{ID_Start}_][\p{ID_Continue}]*\s*=\s*)?$/u.test(before))
    return null;
  const raw = text(index, node),
    quote = raw[0];
  if (!/['"]/.test(quote) || raw.startsWith(quote.repeat(3))) return null;
  let slashes = 0;
  for (let at = raw.length - 2; at >= 1 && raw[at] === "\\"; at--) slashes++;
  const closed = raw.length > 1 && raw.endsWith(quote) && slashes % 2 === 0;
  const contentEnd = node.to - (closed ? 1 : 0);
  if (context.pos < node.from + 1 || context.pos > contentEnd) return null;
  const parameter = call.entry.parameters.find(
    (item) => item.name === call.activeParameter,
  );
  const typed = context.state
    .sliceDoc(node.from + 1, context.pos)
    .toLowerCase();
  const options = parameterLiterals(parameter).flatMap((literal) => {
    const content = stringContent(literal, quote);
    return content === undefined || !content.toLowerCase().startsWith(typed)
      ? []
      : [
          {
            label: content,
            apply: content + (closed ? "" : quote),
            type: "constant",
            detail: parameter?.description,
            boost: 12,
          } satisfies Completion,
        ];
  });
  return options.length
    ? {
        from: node.from + 1,
        to: contentEnd,
        options,
        filter: false,
      }
    : null;
}
function existingValueAhead(context: CompletionContext, call: ActiveCall) {
  const before = context.state.sliceDoc(call.argumentFrom, context.pos);
  if (!/^\s*(?:[\p{ID_Start}_][\p{ID_Continue}]*\s*=\s*)?$/u.test(before))
    return false;
  const line = context.state.doc.lineAt(context.pos),
    after = context.state.sliceDoc(context.pos, line.to).trimStart();
  return /^[+\-'"\d\p{ID_Start}_]/u.test(after);
}
export function editorCompletionSource(context: CompletionContext) {
  if (context.state.readOnly) return null;
  const index = semanticIndex(context.state, context.pos),
    node = index.tree.resolveInner(context.pos, -1);
  const call = getActiveCall(context.state, context.pos);
  if (blocked(node)) {
    const string = plainString(node);
    return string && call
      ? stringCompletions(context, index, string, call)
      : null;
  }
  const line = context.state.doc.lineAt(context.pos),
    prefix = context.state.sliceDoc(line.from, context.pos);
  const member = prefix.match(
    /([\p{ID_Start}_][\p{ID_Continue}]*(?:\s*\.\s*[\p{ID_Start}_][\p{ID_Continue}]*)*)\s*\.\s*([\p{ID_Start}_][\p{ID_Continue}]*)?$/u,
  );
  if (member) {
    const owner = resolveName(
      index,
      scopeAt(index.root, context.pos),
      member[1],
      context.pos,
    )?.canonical;
    if (!owner) return null;
    const canonical = `${owner}.`,
      options: Completion[] = [];
    const seen = new Set<string>();
    for (const entry of Object.values(API_CATALOG)) {
      if (!entry.name.startsWith(canonical)) continue;
      const rest = entry.name.slice(canonical.length),
        label = rest.split(".")[0];
      if (seen.has(label)) continue;
      const direct = API_CATALOG[`${canonical}${label}`];
      if (
        rest.includes(".") &&
        !direct &&
        `${canonical}${label}` !== "openecon.plot"
      )
        continue;
      seen.add(label);
      options.push(
        direct
          ? asCompletion(direct, label)
          : rest.includes(".")
            ? { label, type: "namespace" }
            : asCompletion(entry, label),
      );
    }
    return options.length
      ? {
          from: context.pos - (member[2]?.length ?? 0),
          to: node.name === "PropertyName" ? node.to : context.pos,
          options,
          validFor: /^[\p{ID_Continue}]*$/u,
        }
      : null;
  }
  // An unsupported receiver such as df.head(). must not receive a list of
  // unrelated globals. Receiver inference stays conservative.
  if (/\.\s*[\p{ID_Continue}]*$/u.test(prefix) && !/\d\.\d*$/.test(prefix))
    return null;
  const word = context.matchBefore(/[\p{ID_Start}_][\p{ID_Continue}]*/u);
  const argumentPrefix = call
      ? context.state.sliceDoc(call.argumentFrom, context.pos)
      : "",
    keywordContext = Boolean(
      call && /^\s*[\p{ID_Continue}]*$/u.test(argumentPrefix),
    ),
    valueContext = Boolean(
      call &&
      /^\s*(?:[\p{ID_Start}_][\p{ID_Continue}]*\s*=\s*)?[+\-.\p{ID_Continue}]*$/u.test(
        argumentPrefix,
      ),
    );
  const valueAhead = Boolean(call && existingValueAhead(context, call));
  if (!word && !context.explicit && !keywordContext && !valueContext)
    return null;
  const number = valueContext
    ? context.matchBefore(/[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?/)
    : null;
  const assignmentSuffix = keywordContext
    ? context.state
        .sliceDoc(context.pos, line.to)
        .match(/^[\p{ID_Continue}]*\s*=/u)
    : null;
  const to = assignmentSuffix
    ? context.pos + assignmentSuffix[0].length
    : ["VariableName", "Number", "Boolean", "None"].includes(node.name)
      ? node.to
      : context.pos;
  const sign = valueContext ? context.matchBefore(/[+-]/) : null;
  const from = number?.from ?? word?.from ?? sign?.from ?? context.pos,
    options: Completion[] = [],
    seen = new Set<string>();
  if (valueContext && call && !valueAhead) {
    const parameter = call.entry.parameters.find(
      (item) => item.name === call.activeParameter,
    );
    for (const literal of parameterLiterals(parameter)) {
      options.push({
        label: literal,
        type: "constant",
        detail: parameter?.description,
        boost: literal === "None" ? 1 : 12,
      });
      seen.add(literal);
    }
  }
  if (call && keywordContext) {
    for (const parameter of call.entry.parameters) {
      if (
        ["positional-only", "var-positional", "var-keyword"].includes(
          parameter.kind ?? "",
        ) ||
        call.supplied.includes(parameter.name)
      )
        continue;
      const label = `${parameter.name}=`;
      options.push({
        label,
        type: "property",
        detail: parameter.description,
        boost: parameter.default === undefined ? 10 : 8,
      });
      seen.add(label);
    }
  }
  const currentScope = scopeAt(index.root, context.pos);
  for (
    let scope: Scope | undefined = currentScope;
    scope;
    scope = scope.parent
  ) {
    for (const binding of [...scope.bindings].reverse()) {
      if (
        binding.from > context.pos ||
        seen.has(binding.name) ||
        bindingAt(currentScope, binding.name, context.pos) !== binding
      )
        continue;
      seen.add(binding.name);
      options.push(
        binding.entry && !binding.instance
          ? asCompletion(binding.entry, binding.name)
          : {
              label: binding.name,
              type:
                binding.canonical && !binding.instance
                  ? "namespace"
                  : "variable",
              detail: binding.instance
                ? binding.canonical?.split(".").at(-1)
                : undefined,
              boost: valueContext ? 6 : 3,
            },
      );
    }
  }
  for (const entry of Object.values(API_CATALOG)) {
    if (!entry.name.startsWith("builtins.") && entry.name !== "console.display")
      continue;
    const label = entry.name.split(".").at(-1)!;
    if (seen.has(label) || bindingAt(currentScope, label, context.pos))
      continue;
    seen.add(label);
    options.push({
      ...asCompletion(entry, label),
      boost: valueContext ? -2 : 2,
    });
  }
  let usefulOptions =
    assignmentSuffix || valueAhead
      ? options.filter((option) => option.type === "property")
      : options;
  if (to > context.pos) {
    const typed = context.state.sliceDoc(from, context.pos).toLowerCase();
    usefulOptions = usefulOptions.filter((option) =>
      option.label.toLowerCase().startsWith(typed),
    );
  }
  return usefulOptions.length
    ? {
        from,
        to,
        options: usefulOptions,
        ...(to > context.pos
          ? { filter: false }
          : {
              validFor: number
                ? /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?$/
                : /^[\p{ID_Continue}]*$/u,
            }),
      }
    : null;
}

function pythonCompletionSource(context: CompletionContext) {
  if (context.state.readOnly) return null;
  const index = semanticIndex(context.state, context.pos),
    node = index.tree.resolveInner(context.pos, -1);
  if (blocked(node)) return null;
  const line = context.state.doc.lineAt(context.pos),
    prefix = context.state.sliceDoc(line.from, context.pos);
  if (/\.\s*[\p{ID_Continue}]*$/u.test(prefix) && !/\d\.\d*$/.test(prefix))
    return null;
  const result = globalCompletion(context);
  if (!result || "then" in result) return null;
  const scope = scopeAt(index.root, context.pos),
    call = getActiveCall(context.state, context.pos);
  if (call && existingValueAhead(context, call)) return null;
  if (
    call &&
    /^\s*[\p{ID_Continue}]*$/u.test(
      context.state.sliceDoc(call.argumentFrom, context.pos),
    ) &&
    /^[\p{ID_Continue}]*\s*=/u.test(
      context.state.sliceDoc(context.pos, line.to),
    )
  )
    return null;
  const expressionKeywords: Completion[] = call
    ? ["lambda", "not"].map((label) => ({
        label,
        apply: `${label} `,
        type: "keyword",
        boost: -2,
      }))
    : [];
  for (
    let ancestor: SyntaxNode | null = node;
    call && ancestor;
    ancestor = ancestor.parent
  )
    if (ancestor.name === "FunctionDefinition") {
      if (children(ancestor).some((part) => part.name === "async"))
        expressionKeywords.push({
          label: "await",
          apply: "await ",
          type: "keyword",
          boost: -2,
        });
      break;
    }
  const options = result.options
    .filter(
      (option) =>
        !bindingAt(scope, option.label, context.pos) &&
        (!call || option.type !== "keyword"),
    )
    .map((option) => ({ ...option, boost: call ? -2 : option.boost }));
  const combined: Completion[] = [...options, ...expressionKeywords];
  if (
    ["VariableName", "Number", "Boolean", "None"].includes(node.name) &&
    node.to > context.pos
  ) {
    const typed = context.state
      .sliceDoc(result.from, context.pos)
      .toLowerCase();
    const matching = combined.filter((option) =>
      option.label.toLowerCase().startsWith(typed),
    );
    return matching.length
      ? { from: result.from, to: node.to, options: matching, filter: false }
      : null;
  }
  return combined.length ? { ...result, options: combined } : null;
}

const dismissSignature = StateEffect.define<null>();
const helpFocus = StateEffect.define<boolean>();
const refreshSignature = StateEffect.define<null>();
export const EDITOR_HELP_HIDE_MS = 8000;
const signatureHelp = StateField.define<{
  call: ActiveCall | null;
  dismissed: boolean;
  focused: boolean;
}>({
  create: () => ({ call: null, dismissed: false, focused: false }),
  update(value, transaction) {
    let next = value;
    const input =
      transaction.docChanged ||
      (Boolean(transaction.selection) &&
        !transaction.startState.selection.eq(transaction.state.selection));
    if (
      input ||
      syntaxTree(transaction.startState) !== syntaxTree(transaction.state)
    )
      next = {
        ...next,
        call: getActiveCall(transaction.state),
        dismissed: input ? false : next.dismissed,
      };
    for (const effect of transaction.effects) {
      if (effect.is(dismissSignature)) next = { ...next, dismissed: true };
      if (effect.is(refreshSignature))
        next = { ...next, call: getActiveCall(transaction.startState) };
      if (effect.is(helpFocus))
        next = {
          ...next,
          focused: effect.value,
          call: effect.value ? getActiveCall(transaction.state) : next.call,
        };
    }
    return next;
  },
  provide: (field) =>
    showTooltip.compute([field], (state): Tooltip | null => {
      const value = state.field(field),
        call = value.call;
      if (!value.focused || value.dismissed || !call) return null;
      return {
        pos: state.selection.main.head,
        above: true,
        create: (view) => ({
          dom: helpDOM(call.entry, call.activeParameter),
          mount: () => view.plugin(helpLifetimes)?.onSignatureMounted(),
          destroy: () => view.plugin(helpLifetimes)?.onSignatureUnmounted(),
        }),
      };
    }),
});
const parsedHelpRefresh = ViewPlugin.fromClass(
  class {
    private tree;
    private alive = true;
    private pending = false;
    private view: EditorView;
    constructor(view: EditorView) {
      this.view = view;
      this.tree = syntaxTree(view.state);
    }
    update(update: ViewUpdate) {
      const tree = syntaxTree(update.state);
      const parsed = tree !== this.tree;
      this.tree = tree;
      const focused = update.state.field(signatureHelp).focused;
      if ((parsed && !update.docChanged) || focused !== this.view.hasFocus)
        this.refresh();
    }
    refresh() {
      if (this.pending) return;
      this.pending = true;
      queueMicrotask(() => {
        this.pending = false;
        if (!this.alive) return;
        const focused = this.view.hasFocus;
        if (
          focused ||
          this.view.state.field(signatureHelp).focused !== focused
        ) {
          // Focus notifications may race parser updates; read the current DOM focus.
          this.view.dispatch({
            effects: [helpFocus.of(focused), refreshSignature.of(null)],
          });
        }
      });
    }
    destroy() {
      this.alive = false;
    }
  },
  {
    eventHandlers: {
      focus() {
        this.refresh();
      },
      blur() {
        this.refresh();
      },
    },
  },
);
const functionHover = hoverTooltip(
  (view, pos, side) => {
    const symbol = getEditorSymbol(view.state, pos, side);
    if (!symbol) return null;
    const tooltip: Tooltip = {
      pos: symbol.from,
      end: symbol.to,
      above: true,
      create: () => ({
        dom: helpDOM(symbol.entry),
        mount: () => view.plugin(helpLifetimes)?.onHoverMounted(tooltip),
      }),
    };
    return tooltip;
  },
  { hoverTime: 350, hideOnChange: true },
);

/** Close informational help without changing completion or inline suggestions. */
export function dismissEditorHelp(view: EditorView): void {
  if (!view.state.field(signatureHelp, false)) return;
  view.dispatch({
    effects: [closeHoverTooltip(functionHover), dismissSignature.of(null)],
  });
}

const helpLifetimes = ViewPlugin.fromClass(
  class {
    private hoverIdentity: readonly Tooltip[] = [];
    private hoverTimer: number | undefined;
    private signatureTimer: number | undefined;
    private signatureVisible = false;
    private alive = true;
    private view: EditorView;
    private window: Window;
    private document: Document;
    constructor(view: EditorView) {
      this.view = view;
      this.document = view.dom.ownerDocument;
      this.window = this.document.defaultView!;
      this.document.addEventListener("pointerdown", this.clickAway, true);
    }
    private signatureShown() {
      const help = this.view.state.field(signatureHelp);
      return help.focused && !help.dismissed && Boolean(help.call);
    }
    private clearHover() {
      if (this.hoverTimer !== undefined)
        this.window.clearTimeout(this.hoverTimer);
      this.hoverTimer = undefined;
    }
    private clearSignature() {
      if (this.signatureTimer !== undefined)
        this.window.clearTimeout(this.signatureTimer);
      this.signatureTimer = undefined;
      this.signatureVisible = false;
    }
    onHoverMounted(tooltip: Tooltip) {
      const identity = this.view.state.field(functionHover.active);
      if (!identity.includes(tooltip)) return;
      if (identity === this.hoverIdentity && this.hoverTimer !== undefined)
        return;
      this.clearHover();
      this.hoverIdentity = identity;
      this.hoverTimer = this.window.setTimeout(() => {
        this.hoverTimer = undefined;
        if (
          this.alive &&
          this.view.state.field(functionHover.active) === identity
        )
          this.view.dispatch({ effects: closeHoverTooltip(functionHover) });
      }, EDITOR_HELP_HIDE_MS);
    }
    private restartSignature() {
      this.clearSignature();
      this.signatureVisible = true;
      this.signatureTimer = this.window.setTimeout(() => {
        this.signatureTimer = undefined;
        if (this.alive && this.signatureShown())
          this.view.dispatch({ effects: dismissSignature.of(null) });
      }, EDITOR_HELP_HIDE_MS);
    }
    onSignatureMounted() {
      if (this.signatureShown() && this.signatureTimer === undefined)
        this.restartSignature();
    }
    onSignatureUnmounted() {
      // Parser refreshes replace tooltip DOM; they are not new user activity.
      if (!this.signatureShown()) this.clearSignature();
    }
    update(update: ViewUpdate) {
      const hover = update.state.field(functionHover.active);
      if (hover !== this.hoverIdentity) {
        this.clearHover();
        this.hoverIdentity = hover;
      }
      if (!this.signatureShown()) this.clearSignature();
      else if (
        this.signatureVisible &&
        (update.docChanged ||
          (update.selectionSet &&
            !update.startState.selection.eq(update.state.selection)))
      )
        this.restartSignature();
    }
    private clickAway = (event: Event) => {
      const target = event.target;
      if (
        target instanceof Node &&
        !this.view.dom.contains(target) &&
        !(target instanceof Element && target.closest(".cm-editor-help"))
      )
        dismissEditorHelp(this.view);
    };
    blur() {
      dismissEditorHelp(this.view);
    }
    destroy() {
      this.alive = false;
      this.clearHover();
      this.clearSignature();
      this.document.removeEventListener("pointerdown", this.clickAway, true);
    }
  },
  {
    eventHandlers: {
      blur() {
        this.blur();
      },
    },
  },
);

export function editorHelpExtensions(): Extension[] {
  return [
    autocompletion({
      override: [editorCompletionSource, pythonCompletionSource],
      maxRenderedOptions: 30,
      activateOnTypingDelay: 100,
    }),
    functionHover,
    signatureHelp,
    parsedHelpRefresh,
    helpLifetimes,
    EditorView.focusChangeEffect.of((_state, focusing) =>
      helpFocus.of(focusing),
    ),
    keymap.of([
      {
        key: "F1",
        run(view) {
          const pos = view.state.selection.main.head;
          if (!getEditorSymbol(view.state, pos, -1)) return false;
          activateHover(view, pos, -1, {
            tooltip: functionHover,
            until: (transaction) =>
              transaction.docChanged || Boolean(transaction.selection),
          });
          return true;
        },
      },
      {
        key: "Escape",
        run(view) {
          const value = view.state.field(signatureHelp);
          const hasHover = view.state.field(functionHover.active).length > 0;
          if (!hasHover && (!value.call || value.dismissed || !value.focused))
            return false;
          view.dispatch({
            effects: [closeHoverTooltips, dismissSignature.of(null)],
          });
          return true;
        },
      },
    ]),
    EditorView.theme({
      ".cm-editor-help": {
        padding: "10px 12px",
        maxWidth: "min(480px, calc(100vw - 24px))",
        maxHeight: "min(320px, 60vh)",
        overflow: "auto",
        boxSizing: "border-box",
        fontSize: "12px",
        lineHeight: "1.55",
        color: "#172d4c",
      },
      ".cm-help-signature": {
        fontFamily: '"SFMono-Regular", Consolas, "Liberation Mono", monospace',
        whiteSpace: "pre-wrap",
        overflowWrap: "anywhere",
      },
      ".cm-editor-help p": {
        margin: "7px 0 0",
        fontFamily: "system-ui, sans-serif",
        overflowWrap: "anywhere",
      },
      ".cm-help-active-parameter": {
        fontWeight: "700",
        backgroundColor: "#e7edf6",
        borderRadius: "2px",
        padding: "0 2px",
      },
      ".cm-help-parameter": {
        borderTop: "1px solid #e7edf6",
        paddingTop: "7px",
      },
      ".cm-tooltip-autocomplete .cm-completionDetail": {
        opacity: "0.75",
        maxWidth: "230px",
        overflow: "hidden",
        textOverflow: "ellipsis",
        display: "inline-block",
        verticalAlign: "bottom",
      },
    }),
  ];
}
