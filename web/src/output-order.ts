import type {
  ConsoleOutput,
  ExecutionEvent,
  ExecutionRecord,
} from "./types.ts";

export type OutputTimelineEvent =
  | { type: "stdout"; text: string }
  | { type: "output"; index: number; output: ConsoleOutput };

type RecordedOutput = Pick<ExecutionRecord, "stdout" | "outputs" | "events">;

function validatedEvents(record: RecordedOutput): ExecutionEvent[] | null {
  const events: unknown = record.events;
  if (
    !Array.isArray(events) ||
    record.outputs.length > 20 ||
    events.length > 2 * record.outputs.length + 1
  )
    return null;
  let nextOutput = 0;
  let previousType: "stdout" | "output" | null = null;
  const stdout: string[] = [];
  for (const value of events) {
    if (!value || typeof value !== "object") return null;
    const keys = Object.keys(value);
    if (keys.length !== 2 || !keys.includes("type")) return null;
    if (value.type === "stdout") {
      if (
        !keys.includes("text") ||
        typeof value.text !== "string" ||
        !value.text ||
        previousType === "stdout"
      )
        return null;
      stdout.push(value.text);
    } else if (value.type === "output") {
      if (
        !keys.includes("index") ||
        !Number.isInteger(value.index) ||
        value.index !== nextOutput ||
        value.index >= record.outputs.length
      )
        return null;
      nextOutput++;
    } else return null;
    previousType = value.type;
  }
  return nextOutput === record.outputs.length &&
    stdout.join("") === record.stdout
    ? (events as ExecutionEvent[])
    : null;
}

export function orderedExecutionEvents(
  record: RecordedOutput,
): OutputTimelineEvent[] {
  const events = validatedEvents(record) ?? [
    ...(record.stdout
      ? [{ type: "stdout" as const, text: record.stdout }]
      : []),
    ...record.outputs.map((_, index) => ({ type: "output" as const, index })),
  ];
  const timeline: OutputTimelineEvent[] = [];
  for (const event of events) {
    if (event.type === "output") {
      timeline.push({ ...event, output: record.outputs[event.index] });
    } else if (event.text) {
      timeline.push({ type: "stdout", text: event.text });
    }
  }
  return timeline;
}

export function selectExecutionEvents(
  timeline: OutputTimelineEvent[],
  {
    outputIndex,
    plotsOnly = false,
  }: { outputIndex?: number; plotsOnly?: boolean } = {},
): OutputTimelineEvent[] {
  return timeline.filter((event) => {
    if (event.type === "stdout") return outputIndex === undefined && !plotsOnly;
    return (
      (outputIndex === undefined || event.index === outputIndex) &&
      (!plotsOnly || event.output.type === "plot")
    );
  });
}

export function timelineLatexOutputs(
  timeline: OutputTimelineEvent[],
): ConsoleOutput[] {
  return timeline.map((event) =>
    event.type === "output" ? event.output : { type: "text", data: event.text },
  );
}
