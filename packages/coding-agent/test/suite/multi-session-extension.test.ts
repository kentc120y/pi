import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { noOpUIContext } from "../../src/core/extensions/runner.ts";
import type { ExtensionAPI, SessionHandle } from "../../src/index.ts";
import { cleanupRuntimes, createRuntimeForTest } from "./multi-session-helpers.ts";

describe("pi.sessions controller", () => {
	afterEach(cleanupRuntimes);

	it("delivers a background session's events to handle.on", async () => {
		let pi: ExtensionAPI | undefined;
		// The factory runs for every session created with this runtime, so capture the
		// first (long-lived) session's api only; a captured pi is bound to its session
		// and goes stale once that session is replaced or closed.
		const { runtime, faux } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const handle = await pi!.sessions.create();
		const seenRoles: string[] = [];
		handle.on("message_end", (event) => {
			if (event.message.role === "assistant") seenRoles.push("assistant");
		});

		faux.setResponses([fauxAssistantMessage("hi from background")]);
		await runtime.getSession(handle.id)!.prompt("go");

		expect(seenRoles).toContain("assistant");
	});

	it("delivers handle.on events with the in-session context (pi.on parity)", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime, faux } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const handle = await pi!.sessions.create();
		let sawCtx = false;
		// Unlike the old raw event stream, handle.on now delivers the mapped event plus
		// the target session's ctx, exactly as a same-session pi.on handler receives.
		handle.on("message_end", (_event, ctx) => {
			if (ctx && typeof ctx.cwd === "string") sawCtx = true;
		});

		faux.setResponses([fauxAssistantMessage("hi from background")]);
		await runtime.getSession(handle.id)!.prompt("go");

		expect(sawCtx).toBe(true);
	});

	it("stops delivering handle.on events once the owning session is replaced", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime, faux } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const handle = await pi!.sessions.create();
		let calls = 0;
		handle.on("message_end", () => {
			calls++;
		});

		// Replace the owning (first) session, making its runtime stale. The observer lives
		// on the still-live observed session, so without the staleness guard it would keep
		// firing the gone owner's callback.
		await runtime.newSession();

		faux.setResponses([fauxAssistantMessage("after owner replaced")]);
		await runtime.getSession(handle.id)?.prompt("go");
		expect(calls).toBe(0);
	});

	it("keeps handle.on delivering across a same-id replacement of the observed session", async () => {
		let firstApi: ExtensionAPI | undefined;
		const { runtime, faux } = await createRuntimeForTest((api) => {
			firstApi ??= api;
		});

		// Owner = the long-lived first session (stays alive, just backgrounded below). Observed =
		// a second session, which we later resume from its own file: a same-id replacement that
		// disposes the old object and installs a new one under the same id. An observer bound to
		// the old object would silently stop; keyed by id it must re-home to the new runner.
		const handle = await firstApi!.sessions.create();
		let calls = 0;
		handle.on("message_end", (event) => {
			if (event.message.role === "assistant") calls++;
		});

		// Persist the observed session so it has a resumable file.
		faux.setResponses([fauxAssistantMessage("persist")]);
		await runtime.getSession(handle.id)!.prompt("persist");
		const sessionFile = runtime.getSession(handle.id)!.sessionFile;
		expect(sessionFile).toBeTruthy();
		const callsBeforeReplacement = calls;

		// Focus the observed session (backgrounds the still-alive owner), then resume it from its
		// own file — apply() replaces it under the same id.
		await handle.focus();
		await runtime.switchSession(sessionFile!);
		expect(runtime.session.sessionId).toBe(handle.id);

		faux.setResponses([fauxAssistantMessage("after same-id resume")]);
		await runtime.getSession(handle.id)!.prompt("go again");

		// The observer fired again after the replacement, proving it re-homed to the new runner.
		expect(calls).toBeGreaterThan(callsBeforeReplacement);
	});

	it("routes a rejected async handle.on handler to the error sink, not an unhandled rejection", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime, faux } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		// A background session's bind captures this sink as its onError; set it before
		// creating so the observed session routes handler failures here.
		const errors: string[] = [];
		runtime.setOnBackgroundError((err) => {
			errors.push(err.error);
		});

		const handle = await pi!.sessions.create();
		handle.on("message_end", async () => {
			throw new Error("async handler boom");
		});

		faux.setResponses([fauxAssistantMessage("hi from background")]);
		await runtime.getSession(handle.id)!.prompt("go");
		// Flush microtasks so the observer's rejection .catch runs before asserting.
		await new Promise((resolve) => setTimeout(resolve, 0));

		// Unlike a dropped promise, the rejection surfaces as an extension error on the
		// observed session — parity with how emit() catches awaited pi.on handlers.
		expect(errors.some((message) => message.includes("async handler boom"))).toBe(true);
	});

	it("delivers the final (rewritten) message to handle.on message_end observers", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime, faux } = await createRuntimeForTest((api) => {
			pi ??= api;
			// A message_end handler that rewrites assistant messages. handle.on observers must
			// see this final, persisted message — not the pre-rewrite original.
			api.on("message_end", (event) => {
				if (event.message.role !== "assistant") return undefined;
				return { message: fauxAssistantMessage("REWRITTEN") };
			});
		});

		const textOf = (message: { content: { type: string; text?: string }[] }) =>
			message.content
				.filter((part) => part.type === "text")
				.map((part) => part.text ?? "")
				.join("");

		const handle = await pi!.sessions.create();
		const observed: string[] = [];
		handle.on("message_end", (event) => {
			if (event.message.role === "assistant") observed.push(textOf(event.message));
		});

		faux.setResponses([fauxAssistantMessage("original")]);
		await runtime.getSession(handle.id)!.prompt("go");

		expect(observed).toContain("REWRITTEN");
		expect(observed).not.toContain("original");
	});

	it("fires handle.on lifecycle observers even when the target has no own handler", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const handle = await pi!.sessions.create();
		let shutdownSeen = false;
		// No extension registers a session_shutdown handler, so emitSessionShutdownEvent used to
		// skip emit() (hasHandlers false) and this cross-session observer never fired.
		handle.on("session_shutdown", () => {
			shutdownSeen = true;
		});

		await handle.close();
		expect(shutdownSeen).toBe(true);
		expect(runtime.listSessions().length).toBe(1);
	});

	it("rejects subscribing to non-observable events at the type level", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime } = await createRuntimeForTest((api) => {
			pi ??= api;
		});
		const handle = await pi!.sessions.create();

		// Observable events type-check.
		handle.on("message_end", () => {});
		// Interceptor events are never delivered cross-session, so subscribing must not
		// type-check (otherwise the subscription would silently never fire).
		// @ts-expect-error "context" is not an ObservableSessionEvent type
		handle.on("context", () => {});

		expect(runtime.listSessions().length).toBe(2);
	});

	it("binds a backgrounded session_start non-interactively (prompts take the default, no deadlock)", async () => {
		// A background create binds the session with the no-op UI: a session_start handler that
		// prompts gets the dialog's safe default immediately instead of blocking until focus.
		// This documented limitation lets createSession await the bind without deadlocking.
		let confirmResult: boolean | undefined;
		const { runtime } = await createRuntimeForTest((api) => {
			api.on("session_start", async (_event, ctx) => {
				confirmResult = await ctx.ui.confirm("proceed?", "");
			});
		});

		// Even with a deferring UI factory installed, the startup bind runs before that UI is
		// armed (it is armed only for later turn prompts), so a hanging confirm cannot stall it.
		runtime.setBackgroundUIContextFactory(() => ({
			...noOpUIContext,
			confirm: () => new Promise<boolean>(() => {}),
		}));

		const outcome = await Promise.race([
			runtime.createSession().then(() => "created"),
			new Promise((resolve) => setTimeout(() => resolve("timeout"), 1000)),
		]);
		expect(outcome).toBe("created");
		// session_start ran with the no-op UI → confirm resolved to its default (false), not hung.
		expect(confirmResult).toBe(false);
	});

	it("finishes a backgrounded session's startup before create resolves", async () => {
		let firstApi: ExtensionAPI | undefined;
		let started = false;

		const { faux } = await createRuntimeForTest((api) => {
			if (!firstApi) {
				firstApi = api;
				return;
			}
			// Stand in for slow startup (awaited handlers, resource discovery).
			api.on("session_start", async () => {
				await new Promise((resolve) => setTimeout(resolve, 10));
				started = true;
			});
		});

		// createSession awaits the (non-interactive) background bind, so session_start has fully
		// run by the time the handle exists — no readiness gate is needed before the first prompt,
		// and the handle never reports a half-started state.
		const handle = await firstApi!.sessions.create();
		expect(started).toBe(true);
		expect(handle.status).toBe("idle");

		faux.setResponses([fauxAssistantMessage("ok")]);
		await handle.sendUserMessage("go");
	});

	it("aborts an outer focus when a session_blur handler re-enters and moves focus", async () => {
		let firstApi: ExtensionAPI | undefined;
		let thirdHandle: SessionHandle | undefined;
		let blurReentered = false;

		const { runtime } = await createRuntimeForTest((api) => {
			if (firstApi) return;
			firstApi = api;
			// When the first session is backgrounded, re-enter focus to a third session. The
			// guard (run once) keeps the re-entrant blur from recursing into itself.
			api.on("session_blur", async () => {
				if (!thirdHandle || blurReentered) return;
				blurReentered = true;
				await thirdHandle.focus();
			});
		});

		const second = await firstApi!.sessions.create();
		const third = await firstApi!.sessions.create();
		thirdHandle = third;

		// Focusing `second` blurs the first session; its blur handler focuses `third` inline.
		// The outer focus must abort rather than overwrite that, leaving `third` focused.
		await second.focus();
		expect(runtime.session.sessionId).toBe(third.id);
		expect(blurReentered).toBe(true);
	});

	it("reports a closed session's handle status as closed, not idle", async () => {
		let pi: ExtensionAPI | undefined;
		const { runtime } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const handle = await pi!.sessions.create();
		expect(handle.status).toBe("idle");

		await handle.close();
		// A stale handle must not masquerade as a live idle session.
		expect(handle.status).toBe("closed");
		expect(runtime.listSessions().length).toBe(1);
	});

	it("rejects sendUserMessage for a closed session", async () => {
		let firstApi: ExtensionAPI | undefined;
		await createRuntimeForTest((api) => {
			firstApi ??= api;
		});

		const handle = await firstApi!.sessions.create();
		await handle.close();
		// The session is gone; sending against a stale handle must reject rather than drive a
		// disposed session.
		await expect(handle.sendUserMessage("go")).rejects.toThrow("Session not found");
	});

	it("arms an attended background session's turn prompts with the deferring UI; autonomous uses no-op", async () => {
		const { runtime } = await createRuntimeForTest(() => {});
		runtime.setBackgroundUIContextFactory(() => ({ ...noOpUIContext }));

		const attended = await runtime.createSession({ focus: false });
		const autonomous = await runtime.createSession({ focus: false, autonomous: true });

		// An attended background session is bound non-interactively for startup, then armed with
		// the deferring UI (mode "tui", hasUI true) so its turn prompts block & surface on focus.
		const attendedRunner = runtime.getSession(attended)!.extensionRunner;
		expect(attendedRunner.hasUI()).toBe(true);
		expect(attendedRunner.createContext().mode).toBe("tui");

		// An autonomous background session keeps the no-op UI so its turn prompts proceed with a
		// safe default instead of waiting for a user who may never focus it.
		const autonomousRunner = runtime.getSession(autonomous)!.extensionRunner;
		expect(autonomousRunner.hasUI()).toBe(false);
		expect(autonomousRunner.createContext().mode).toBe("print");
	});

	it("cleans up host state when a resume recreates the focused session under the same id", async () => {
		const { runtime, faux } = await createRuntimeForTest(() => {});
		const focusedId = runtime.session.sessionId;

		faux.setResponses([fauxAssistantMessage("persisted")]);
		await runtime.session.prompt("write the session file");
		const sessionFile = runtime.session.sessionFile;
		expect(sessionFile).toBeTruthy();

		const closed: string[] = [];
		runtime.setOnSessionClosed((id) => closed.push(id));

		// Resuming the focused session's own file disposes and recreates it under the same
		// id; the disposed object's retained host state must still be cleaned up.
		await runtime.switchSession(sessionFile!);

		expect(runtime.session.sessionId).toBe(focusedId);
		expect(closed).toContain(focusedId);
	});

	it("exposes pi.sessions to create, list, focus, and close sessions", async () => {
		let pi: ExtensionAPI | undefined;
		// Capture the first session's api only (see note above): the test drives the
		// session lifecycle through this original, always-live session's pi.sessions.
		const { runtime } = await createRuntimeForTest((api) => {
			pi ??= api;
		});

		const firstId = runtime.session.sessionId;
		expect(pi!.sessions.list().map((h) => h.id)).toEqual([firstId]);
		expect(pi!.sessions.focused.id).toBe(firstId);

		const handle = await pi!.sessions.create();
		expect(pi!.sessions.list().length).toBe(2);
		expect(pi!.sessions.focused.id).toBe(firstId); // create does not focus

		await handle.focus();
		expect(pi!.sessions.focused.id).toBe(handle.id);
		expect(runtime.session.sessionId).toBe(handle.id);

		await handle.close();
		expect(pi!.sessions.list().length).toBe(1);
		expect(pi!.sessions.focused.id).toBe(firstId);
	});
});
