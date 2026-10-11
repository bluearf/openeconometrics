import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import ts from "typescript";
import { JSDOM } from "jsdom";
const dom = new JSDOM("<!doctype html><body></body>", {
  url: "http://localhost",
  pretendToBeVisual: true,
});
for (const name of [
  "window",
  "document",
  "navigator",
  "HTMLElement",
  "HTMLInputElement",
  "Event",
  "MouseEvent",
])
  Object.defineProperty(globalThis, name, {
    value: dom.window[name as keyof typeof dom.window],
    configurable: true,
  });
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
const React = await import("react");
const { createRoot } = await import("react-dom/client");
const require = createRequire(import.meta.url);
const output = ts
  .transpileModule(
    readFileSync(new URL("../src/EmailCodeForm.tsx", import.meta.url), "utf8"),
    {
      compilerOptions: {
        jsx: ts.JsxEmit.ReactJSX,
        module: ts.ModuleKind.ESNext,
      },
    },
  )
  .outputText.replace(
    /from (["'])(react(?:\/jsx-runtime)?)\1/g,
    (_match, _quote, name) =>
      "from " + JSON.stringify(pathToFileURL(require.resolve(name)).href),
  );
const { default: Form } = await import(
  "data:text/javascript;base64," + Buffer.from(output).toString("base64")
);
async function harness(
  options: {
    available?: boolean;
    sendError?: boolean;
    verifyError?: boolean;
  } = {},
) {
  const calls: string[] = [];
  const session = {
    cancel() {
      calls.push("cancel");
    },
    async send(email: string) {
      calls.push("send:" + email);
      if (options.sendError) throw new Error("Delivery unavailable");
      return 60;
    },
    async verify(code: string) {
      calls.push("verify:" + code);
      if (options.verifyError) throw new Error("Incorrect code");
    },
  };
  const holder = document.createElement("div");
  document.body.append(holder);
  const root = createRoot(holder);
  await React.act(async () =>
    root.render(
      React.createElement(Form, {
        session,
        available: options.available !== false,
        disabled: false,
        onBusy: (busy: boolean) => calls.push("busy:" + busy),
      }),
    ),
  );
  const edit = async (label: string, value: string) =>
    React.act(async () => {
      const parent = [...holder.querySelectorAll("label")].find(
        (item) => item.textContent === label,
      )!;
      assert.ok(parent, label);
      const input = parent.querySelector("input")!;
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  const submit = () =>
    React.act(async () =>
      holder
        .querySelector("form")!
        .dispatchEvent(
          new Event("submit", { bubbles: true, cancelable: true }),
        ),
    );
  const click = (name: string) =>
    React.act(async () => {
      const button = [...holder.querySelectorAll("button")].find(
        (item) => item.textContent === name,
      )!;
      assert.ok(button, name);
      button.click();
    });
  const close = async () => {
    await React.act(async () => root.unmount());
    holder.remove();
  };
  return { holder, calls, edit, submit, click, close };
}
test("email UI advances only after delivery and verifies a code without any password or account-link step", async () => {
  const h = await harness();
  try {
    assert.equal(h.holder.querySelectorAll('input[type="password"]').length, 0);
    await h.edit("Email", "person@example.test");
    await h.submit();
    assert.match(
      h.holder.textContent!,
      /six-digit code sent to person@example.test/,
    );
    assert.ok(h.calls.includes("send:person@example.test"));
    const resend = [...h.holder.querySelectorAll("button")].find((item) =>
      /Resend in/.test(item.textContent!),
    )!;
    assert.equal(resend.disabled, true);
    await h.edit("Verification code", "0a12345");
    await h.submit();
    assert.ok(h.calls.includes("verify:012345"));
    assert.deepEqual(
      h.calls.filter((value) => value.startsWith("busy:")),
      ["busy:true", "busy:false", "busy:true", "busy:false"],
    );
    await h.click("Use another email");
    assert.equal(
      h.holder.querySelectorAll('input[autocomplete="one-time-code"]').length,
      0,
    );
  } finally {
    await h.close();
  }
  assert.ok(h.calls.includes("cancel"));
});
test("failed mail delivery and wrong codes remain visible failures", async () => {
  for (const mode of ["send", "verify"]) {
    const h = await harness({
      sendError: mode === "send",
      verifyError: mode === "verify",
    });
    try {
      await h.edit("Email", "person@example.test");
      await h.submit();
      if (mode === "send") {
        assert.match(
          h.holder.querySelector('[role="alert"]')!.textContent!,
          /Delivery unavailable/,
        );
        assert.equal(
          h.holder.querySelectorAll('input[autocomplete="one-time-code"]')
            .length,
          0,
        );
      } else {
        await h.edit("Verification code", "123456");
        await h.submit();
        assert.match(
          h.holder.querySelector('[role="alert"]')!.textContent!,
          /Incorrect code/,
        );
        assert.ok(h.calls.includes("verify:123456"));
      }
    } finally {
      await h.close();
    }
  }
});
test("an unconfigured sender disables email delivery and does not claim a code was sent", async () => {
  const h = await harness({ available: false });
  try {
    await h.edit("Email", "person@example.test");
    await h.submit();
    assert.match(h.holder.textContent!, /temporarily unavailable/);
    assert.equal(
      h.calls.filter((value) => value.startsWith("send:")).length,
      0,
    );
    assert.equal(h.holder.querySelector("button")!.disabled, true);
  } finally {
    await h.close();
  }
});
