import assert from "node:assert/strict";
import test from "node:test";
import { latexDocument } from "../src/latex-export.ts";
import {
  orderedExecutionEvents,
  selectExecutionEvents,
  timelineLatexOutputs,
} from "../src/output-order.ts";
import type { ConsoleOutput, ExecutionEvent } from "../src/types.ts";

const outputs: ConsoleOutput[] = [
  {
    type: "table",
    data: { columns: ["value"], rows: [[1]] },
    latex: String.raw`\noindent table-output`,
  },
  { type: "model", data: {}, latex: String.raw`\noindent model-output` },
  { type: "plot", data: {}, latex: String.raw`\noindent plot-output` },
];

function record() {
  return {
    stdout: "before\nbetween\nafter\n",
    outputs,
    events: [
      { type: "stdout", text: "before\n" },
      { type: "output", index: 0 },
      { type: "stdout", text: "between\n" },
      { type: "output", index: 1 },
      { type: "output", index: 2 },
      { type: "stdout", text: "after\n" },
    ] as ExecutionEvent[],
  };
}

test("stdout, tables, models and charts stay in the actual execution sequence", () => {
  const input = record();
  const unchanged = JSON.stringify(input);
  const timeline = orderedExecutionEvents(input);
  assert.deepEqual(
    timeline.map((event) =>
      event.type === "stdout" ? event.text : event.output.type,
    ),
    ["before\n", "table", "between\n", "model", "plot", "after\n"],
  );
  assert.equal(JSON.stringify(input), unchanged);
  assert.equal(
    timeline.filter((event) => event.type === "output").length,
    outputs.length,
  );
});

test("prints after displayed results stay after them in the coalesced stdout block", () => {
  const timeline = orderedExecutionEvents({
    stdout: "hello\nhello2\n",
    outputs,
    events: [
      { type: "output", index: 0 },
      { type: "output", index: 1 },
      { type: "output", index: 2 },
      { type: "stdout", text: "hello\nhello2\n" },
    ],
  });
  assert.deepEqual(
    timeline.map((event) =>
      event.type === "stdout" ? event.text : event.output.type,
    ),
    ["table", "model", "plot", "hello\nhello2\n"],
  );
});

test("plot filtering and a single selected output exclude unrelated printed text", () => {
  const timeline = orderedExecutionEvents(record());
  assert.deepEqual(
    selectExecutionEvents(timeline, { plotsOnly: true }).map(
      (event) => event.type === "output" && event.index,
    ),
    [2],
  );
  assert.deepEqual(
    selectExecutionEvents(timeline, { outputIndex: 1 }).map(
      (event) => event.type === "output" && event.index,
    ),
    [1],
  );
  assert.deepEqual(
    selectExecutionEvents(timeline, { plotsOnly: true, outputIndex: 1 }),
    [],
  );
});

test("old history preserves the aggregate stdout-first fallback", () => {
  const timeline = orderedExecutionEvents({ stdout: "old\n", outputs });
  assert.deepEqual(
    timeline.map((event) =>
      event.type === "stdout" ? event.text : event.output.type,
    ),
    ["old\n", "table", "model", "plot"],
  );
});

test("malformed or incomplete event streams fall back without losing or duplicating outputs", () => {
  const invalid: unknown[] = [
    null,
    {},
    [null],
    [{ type: "unknown" }],
    [{ type: "stdout", text: 1 }],
    [{ type: "stdout", text: "" }],
    [{ type: "stdout", text: "fallback\n", extra: true }],
    [
      { type: "stdout", text: "fall" },
      { type: "stdout", text: "back\n" },
      ...outputs.map((_, index) => ({ type: "output", index })),
    ],
    [{ type: "output", index: -1 }],
    [{ type: "output", index: 3 }],
    [{ type: "output", index: 0.5 }],
    [{ type: "output", index: true }],
    [{ type: "output", index: 0, extra: true }],
    [
      { type: "output", index: 1 },
      { type: "output", index: 0 },
    ],
    [
      { type: "output", index: 0 },
      { type: "output", index: 0 },
    ],
    [{ type: "output", index: 0 }],
    [
      { type: "stdout", text: "wrong text" },
      ...outputs.map((_, index) => ({ type: "output", index })),
    ],
  ];
  for (const events of invalid) {
    const timeline = orderedExecutionEvents({
      stdout: "fallback\n",
      outputs,
      events: events as ExecutionEvent[],
    });
    assert.deepEqual(
      timeline.map((event) =>
        event.type === "stdout" ? event.text : event.output.type,
      ),
      ["fallback\n", "table", "model", "plot"],
    );
  }
});

test("LaTeX uses the same ordered stream and escapes printed text instead of executing it", () => {
  const input = record();
  input.stdout += String.raw`\input{private} & %`;
  input.events[input.events.length - 1] = {
    type: "stdout",
    text: "after\n" + String.raw`\input{private} & %`,
  };
  const timeline = orderedExecutionEvents(input);
  const source = latexDocument(timelineLatexOutputs(timeline));
  const expected = [
    "before",
    "table-output",
    "between",
    "model-output",
    "plot-output",
    "after",
  ];
  const positions = expected.map((label) => source.indexOf(label));
  assert.ok(
    positions.every(
      (position, index) =>
        position >= 0 && (!index || position > positions[index - 1]),
    ),
  );
  assert.ok(
    source.includes(String.raw`\textbackslash{}input\{private\} \& \%`),
  );
  assert.ok(!source.includes(String.raw`\input{private}`));
  assert.equal((source.match(/before/g) ?? []).length, 1);
});

test("stdout-only and empty executions have an exportable text stream or no events", () => {
  const timeline = orderedExecutionEvents({
    stdout: "printed only\n",
    outputs: [],
    events: [{ type: "stdout", text: "printed only\n" }],
  });
  assert.ok(
    latexDocument(timelineLatexOutputs(timeline)).includes("printed only"),
  );
  assert.deepEqual(
    orderedExecutionEvents({ stdout: "", outputs: [], events: [] }),
    [],
  );
});
