/** Pure coordination primitives for the editor's asynchronous persistence. */

export type SaveStatus = "saved" | "saving" | "error";

export interface AutosaverOptions {
  initialValue: string;
  save: (value: string) => Promise<unknown>;
  delayMs?: number;
  onStatus?: (status: SaveStatus, error?: unknown) => void;
}

/**
 * Debounce before the first write, then serialize writes of the latest value.
 *
 * Never compare against an old acknowledged value while another write is in
 * flight: that write may replace it. An uncertain failed write also invalidates
 * the acknowledged value, so undoing an edit cannot accidentally skip a repair.
 */
export class LatestValueAutosaver {
  private desiredValue: string;
  private savedValue: string | undefined;
  private readonly save: AutosaverOptions["save"];
  private readonly onStatus: NonNullable<AutosaverOptions["onStatus"]>;
  private readonly delayMs: number;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private operation: Promise<void> | null = null;
  private disposed = false;

  constructor(options: AutosaverOptions) {
    this.desiredValue = options.initialValue;
    this.savedValue = options.initialValue;
    this.save = options.save;
    this.onStatus = options.onStatus ?? (() => {});
    this.delayMs = options.delayMs ?? 650;
    if (!Number.isFinite(this.delayMs) || this.delayMs < 0) {
      throw new RangeError("Autosave delay must be finite and nonnegative.");
    }
  }

  /** The last definitely acknowledged value, not the current editor text. */
  getSavedValue(): string | undefined {
    return this.savedValue;
  }

  update(value: string): void {
    if (this.disposed) return;
    this.desiredValue = value;
    this.clearTimer();
    if (this.operation) {
      // The drain loop must observe even an undo back to savedValue.
      this.onStatus("saving");
      return;
    }
    if (value === this.savedValue) {
      this.onStatus("saved");
      return;
    }
    this.onStatus("saving");
    this.timer = setTimeout(() => {
      this.timer = null;
      // Errors are surfaced through onStatus; timer callbacks must not leak
      // unhandled rejections. Explicit flush() callers can await the error.
      void this.start().catch(() => {});
    }, this.delayMs);
  }

  /** Flush the current latest value, also draining edits made during its write. */
  flush(): Promise<void> {
    this.clearTimer();
    return this.start();
  }

  /** Stop scheduled work and callbacks; an already sent request cannot be undone. */
  dispose(): void {
    this.disposed = true;
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private start(): Promise<void> {
    if (this.disposed) return Promise.resolve();
    if (this.operation) return this.operation;
    if (this.desiredValue === this.savedValue) {
      this.onStatus("saved");
      return Promise.resolve();
    }
    // Establish operation before invoking save/onStatus, including synchronous
    // callbacks, so update()/flush() cannot start an overlapping write.
    const operation = Promise.resolve().then(() => this.drain());
    this.operation = operation;
    this.onStatus("saving");
    void operation.then(
      () => {
        this.operation = null;
        if (this.disposed) return;
        if (this.desiredValue === this.savedValue) this.onStatus("saved");
        else this.update(this.desiredValue);
      },
      (error: unknown) => {
        this.operation = null;
        if (!this.disposed) this.onStatus("error", error);
      },
    );
    return operation;
  }

  private async drain(): Promise<void> {
    while (!this.disposed && this.desiredValue !== this.savedValue) {
      const target = this.desiredValue;
      try {
        await this.save(target);
        this.savedValue = target;
      } catch (error) {
        // A lost response may still mean the server wrote target. Do not claim
        // that the formerly saved value remains on disk after a failed request.
        this.savedValue = undefined;
        if (!this.disposed && this.desiredValue !== target) continue;
        throw error;
      }
    }
  }
}

export interface ResponseTicket {
  readonly epoch: number;
  readonly sequence: number;
}

/** Reject responses from previous actions and out-of-order status snapshots. */
export class ResponseGate {
  private epoch = 0;
  private issued = 0;
  private applied = 0;

  issue(): ResponseTicket {
    return { epoch: this.epoch, sequence: ++this.issued };
  }

  /** Call when starting execute/reset/interrupt, before any new state request. */
  invalidate(): void {
    this.epoch += 1;
    this.issued = 0;
    this.applied = 0;
  }

  accept(ticket: ResponseTicket): boolean {
    if (
      ticket.epoch !== this.epoch ||
      ticket.sequence <= this.applied ||
      ticket.sequence > this.issued
    ) {
      return false;
    }
    this.applied = ticket.sequence;
    return true;
  }
}
