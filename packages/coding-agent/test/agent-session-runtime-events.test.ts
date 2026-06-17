import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, registerFauxProvider } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import {
	type CreateAgentSessionRuntimeFactory,
	createAgentSessionFromServices,
	createAgentSessionRuntime,
	createAgentSessionServices,
} from "../src/core/agent-session-runtime.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import type {
	ExtensionFactory,
	ExtensionUIContext,
	SessionBeforeForkEvent,
	SessionBeforeSwitchEvent,
	SessionShutdownEvent,
	SessionStartEvent,
} from "../src/index.ts";

type RecordedSessionEvent =
	| SessionBeforeSwitchEvent
	| SessionBeforeForkEvent
	| SessionShutdownEvent
	| SessionStartEvent;

describe("AgentSessionRuntime session lifecycle events", () => {
	const cleanups: Array<() => Promise<void> | void> = [];

	afterEach(async () => {
		while (cleanups.length > 0) {
			await cleanups.pop()?.();
		}
	});

	async function createRuntimeHost(extensionFactory: ExtensionFactory) {
		const tempDir = join(tmpdir(), `pi-runtime-events-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });

		const faux = registerFauxProvider();
		faux.setResponses([fauxAssistantMessage("one"), fauxAssistantMessage("two"), fauxAssistantMessage("three")]);

		const authStorage = AuthStorage.inMemory();
		authStorage.setRuntimeApiKey(faux.getModel().provider, "faux-key");

		const runtimeOptions = {
			agentDir: tempDir,
			authStorage,
			model: faux.getModel(),
			resourceLoaderOptions: {
				extensionFactories: [extensionFactory],
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
			},
		};
		const createRuntime: CreateAgentSessionRuntimeFactory = async ({ cwd, sessionManager, sessionStartEvent }) => {
			const services = await createAgentSessionServices({
				...runtimeOptions,
				cwd,
			});
			return {
				...(await createAgentSessionFromServices({
					services,
					sessionManager,
					sessionStartEvent,
					model: faux.getModel(),
				})),
				services,
				diagnostics: services.diagnostics,
			};
		};
		const runtimeHost = await createAgentSessionRuntime(createRuntime, {
			cwd: tempDir,
			agentDir: tempDir,
			sessionManager: SessionManager.create(tempDir),
		});
		await runtimeHost.session.bindExtensions({});

		cleanups.push(async () => {
			await runtimeHost.dispose();
			faux.unregister();
			if (existsSync(tempDir)) {
				rmSync(tempDir, { recursive: true, force: true });
			}
		});

		return { runtimeHost, faux };
	}

	it("emits session_before_switch and session_start for new and resume flows", async () => {
		const events: RecordedSessionEvent[] = [];
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_before_switch", (event) => {
				events.push(event);
			});
			pi.on("session_shutdown", (event) => {
				events.push(event);
			});
			pi.on("session_start", (event) => {
				events.push(event);
			});
		});

		expect(events).toEqual([{ type: "session_start", reason: "startup" }]);
		events.length = 0;

		await runtimeHost.session.prompt("hello");
		const originalSessionFile = runtimeHost.session.sessionFile;
		expect(originalSessionFile).toBeTruthy();

		const newSessionResult = await runtimeHost.newSession();
		expect(newSessionResult.cancelled).toBe(false);
		await runtimeHost.session.bindExtensions({});
		const secondSessionFile = runtimeHost.session.sessionFile;
		expect(events).toEqual([
			{ type: "session_before_switch", reason: "new", targetSessionFile: undefined },
			{ type: "session_shutdown", reason: "new", targetSessionFile: secondSessionFile },
			{ type: "session_start", reason: "new", previousSessionFile: originalSessionFile },
		]);

		events.length = 0;
		expect(secondSessionFile).toBeTruthy();

		const switchResult = await runtimeHost.switchSession(originalSessionFile!);
		expect(switchResult.cancelled).toBe(false);
		await runtimeHost.session.bindExtensions({});
		expect(events).toEqual([
			{ type: "session_before_switch", reason: "resume", targetSessionFile: originalSessionFile },
			{ type: "session_shutdown", reason: "resume", targetSessionFile: originalSessionFile },
			{ type: "session_start", reason: "resume", previousSessionFile: secondSessionFile },
		]);
	});

	it("honors session_before_switch cancellation", async () => {
		const events: RecordedSessionEvent[] = [];
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_before_switch", (event) => {
				events.push(event);
				return { cancel: true };
			});
			pi.on("session_start", (event) => {
				events.push(event);
			});
		});

		expect(events).toEqual([{ type: "session_start", reason: "startup" }]);
		events.length = 0;

		await runtimeHost.session.prompt("hello");
		const originalSessionFile = runtimeHost.session.sessionFile;

		const result = await runtimeHost.newSession();
		expect(result.cancelled).toBe(true);
		expect(runtimeHost.session.sessionFile).toBe(originalSessionFile);
		expect(events).toEqual([{ type: "session_before_switch", reason: "new", targetSessionFile: undefined }]);
	});

	it("runs beforeSessionInvalidate after session_shutdown and before rebindSession", async () => {
		const phases: string[] = [];
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_shutdown", () => {
				phases.push("session_shutdown");
			});
		});
		const oldSession = runtimeHost.session;
		runtimeHost.setBeforeSessionInvalidate(() => {
			phases.push("beforeSessionInvalidate");
			expect(oldSession.extensionRunner.createContext().cwd).toBe(oldSession.sessionManager.getCwd());
		});
		runtimeHost.setRebindSession(async () => {
			phases.push("rebindSession");
		});

		await runtimeHost.newSession();

		expect(phases).toEqual(["session_shutdown", "beforeSessionInvalidate", "rebindSession"]);
		expect(() => oldSession.extensionRunner.createContext().cwd).toThrow(
			"This extension ctx is stale after session replacement or reload. Do not use a captured pi or command ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For reload, do not use the old ctx after await ctx.reload().",
		);
		runtimeHost.setBeforeSessionInvalidate(undefined);
		runtimeHost.setRebindSession(undefined);
	});

	it("emits session_before_fork and session_start and honors cancellation", async () => {
		const events: RecordedSessionEvent[] = [];
		let cancelNextFork = false;
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_before_fork", (event) => {
				events.push(event);
				if (cancelNextFork) {
					cancelNextFork = false;
					return { cancel: true };
				}
			});
			pi.on("session_shutdown", (event) => {
				events.push(event);
			});
			pi.on("session_start", (event) => {
				events.push(event);
			});
		});

		expect(events).toEqual([{ type: "session_start", reason: "startup" }]);
		events.length = 0;

		await runtimeHost.session.prompt("hello");
		const userMessage = runtimeHost.session.getUserMessagesForForking()[0];
		const previousSessionFile = runtimeHost.session.sessionFile;

		const successResult = await runtimeHost.fork(userMessage.entryId);
		expect(successResult.cancelled).toBe(false);
		expect(successResult.selectedText).toBe("hello");
		await runtimeHost.session.bindExtensions({});
		expect(events).toEqual([
			{ type: "session_before_fork", entryId: userMessage.entryId, position: "before" },
			{ type: "session_shutdown", reason: "fork", targetSessionFile: runtimeHost.session.sessionFile },
			{ type: "session_start", reason: "fork", previousSessionFile },
		]);

		events.length = 0;
		cancelNextFork = true;
		const cancelResult = await runtimeHost.fork(userMessage.entryId);
		expect(cancelResult).toEqual({ cancelled: true });
		expect(events).toEqual([{ type: "session_before_fork", entryId: userMessage.entryId, position: "before" }]);

		events.length = 0;
		cancelNextFork = true;
		const cancelAtResult = await runtimeHost.fork("missing-entry", { position: "at" });
		expect(cancelAtResult).toEqual({ cancelled: true });
		expect(events).toEqual([{ type: "session_before_fork", entryId: "missing-entry", position: "at" }]);
	});

	it("emits session_start only once per session across repeated binds (focus re-attach)", async () => {
		const events: RecordedSessionEvent[] = [];
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_start", (event) => {
				events.push(event);
			});
		});

		expect(events).toEqual([{ type: "session_start", reason: "startup" }]);

		// Re-binding the same session (what focusing it does in the TUI) re-applies the
		// UI bindings but must NOT replay session_start or re-run resource discovery.
		await runtimeHost.session.bindExtensions({});
		await runtimeHost.session.bindExtensions({});

		expect(events).toEqual([{ type: "session_start", reason: "startup" }]);
	});

	it("focuses an already-live session instead of opening a duplicate on resume", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});

		await runtimeHost.session.prompt("hello");
		const firstId = runtimeHost.session.sessionId;
		const firstFile = runtimeHost.session.sessionFile;
		expect(firstFile).toBeTruthy();

		// Open a second session in the background and focus it.
		const secondId = await runtimeHost.createSession({ focus: true });
		expect(runtimeHost.listSessions().length).toBe(2);
		expect(runtimeHost.session.sessionId).toBe(secondId);
		expect(secondId).not.toBe(firstId);

		// Resuming the first session, which is already live, must focus it rather than
		// open a second live copy that would double-write the same session file.
		const result = await runtimeHost.switchSession(firstFile!);
		expect(result.cancelled).toBe(false);
		expect(runtimeHost.listSessions().length).toBe(2);
		expect(runtimeHost.session.sessionId).toBe(firstId);
	});

	it("invokes onSessionClosed with the closed session id so the host can prune per-session state", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const closed: string[] = [];
		runtimeHost.setOnSessionClosed((id) => {
			closed.push(id);
		});

		const firstId = runtimeHost.session.sessionId;
		const secondId = await runtimeHost.createSession({ focus: false });
		expect(runtimeHost.listSessions().length).toBe(2);

		await runtimeHost.closeSession(secondId);

		expect(closed).toEqual([secondId]);
		expect(runtimeHost.listSessions().length).toBe(1);
		expect(runtimeHost.session.sessionId).toBe(firstId);
	});

	it("invokes onSessionClosed for the replaced session when a new session replaces it", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const closed: string[] = [];
		runtimeHost.setOnSessionClosed((id) => {
			closed.push(id);
		});

		const firstId = runtimeHost.session.sessionId;
		// /new disposes the focused session and replaces it (it is not a background
		// close), so the prune hook must still fire for the session that went away.
		const result = await runtimeHost.newSession();
		expect(result.cancelled).toBe(false);

		expect(closed).toEqual([firstId]);
		expect(runtimeHost.session.sessionId).not.toBe(firstId);
	});

	it("detachUI drops the session's real UI so a backgrounded session cannot draw", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const runner = runtimeHost.session.extensionRunner;

		// Attach a real UI context and TUI mode, as focusing the session does.
		await runtimeHost.session.bindExtensions({ uiContext: {} as unknown as ExtensionUIContext, mode: "tui" });
		expect(runner.hasUI()).toBe(true);
		expect(runner.createContext().mode).toBe("tui");

		// Losing focus detaches it; the runner falls back to its no-op UI, and mode is
		// downgraded so extensions gating on `mode === "tui"` take the non-interactive
		// path instead of calling into the no-op UI.
		runtimeHost.session.detachUI();
		expect(runner.hasUI()).toBe(false);
		expect(runner.createContext().mode).toBe("print");
	});

	it("fires session_start in TUI mode for a focus-created session", async () => {
		let inTuiRebind = false;
		const firedDuringTuiRebind = new Map<string, boolean>();
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_start", (event) => {
				firedDuringTuiRebind.set(event.reason, inTuiRebind);
			});
		});

		// Simulate the TUI host: focusing a session re-binds it with a real UI context
		// in "tui" mode, exactly like bindCurrentSessionExtensions() does.
		runtimeHost.setRebindSession(async (session) => {
			inTuiRebind = true;
			await session.bindExtensions({ uiContext: {} as unknown as ExtensionUIContext, mode: "tui" });
			inTuiRebind = false;
		});
		firedDuringTuiRebind.clear(); // discard the startup session's session_start

		const newId = await runtimeHost.createSession({ focus: true });
		const runner = runtimeHost.getSession(newId)?.extensionRunner;

		// A focus-created session must fire its one-time session_start during the TUI
		// rebind (real UI attached), not during a throwaway print-mode pre-bind that the
		// once-only gate would then make the TUI rebind unable to replay.
		expect(firedDuringTuiRebind.get("new")).toBe(true);
		expect(runner?.hasUI()).toBe(true);
		expect(runner?.createContext().mode).toBe("tui");
	});

	it("detachUI clears the focus-local abort and command-context handlers", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const session = runtimeHost.session;
		const runner = session.extensionRunner;

		let abortHandlerCalls = 0;
		let newSessionHandlerCalls = 0;
		// Bind as focusing the session does: real UI plus the InteractiveMode closures
		// that act on whatever session is focused.
		await session.bindExtensions({
			uiContext: {} as unknown as ExtensionUIContext,
			mode: "tui",
			abortHandler: () => {
				abortHandlerCalls++;
			},
			commandContextActions: {
				waitForIdle: async () => {},
				newSession: async () => {
					newSessionHandlerCalls++;
					return { cancelled: false };
				},
				fork: async () => ({ cancelled: false }),
				navigateTree: async () => ({ cancelled: false }),
				switchSession: async () => ({ cancelled: false }),
				reload: async () => {},
			},
		});

		// While focused, the context routes through those closures.
		runner.createContext().abort();
		await runner.createCommandContext().newSession();
		expect(abortHandlerCalls).toBe(1);
		expect(newSessionHandlerCalls).toBe(1);

		// Backgrounding must drop both, so an event/handle.on callback firing on this
		// now-background session can no longer drive the session focused now (and
		// ctx.abort() no longer skips aborting this session itself).
		session.detachUI();
		runner.createContext().abort();
		await runner.createCommandContext().newSession();
		expect(abortHandlerCalls).toBe(1);
		expect(newSessionHandlerCalls).toBe(1);
	});

	it("serializes concurrent focus operations so their rebinds do not interleave", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const b1 = await runtimeHost.createSession({ focus: false });
		const b2 = await runtimeHost.createSession({ focus: false });

		const order: string[] = [];
		runtimeHost.setRebindSession(async (session) => {
			order.push(`start:${session.sessionId}`);
			await Promise.resolve();
			await Promise.resolve();
			order.push(`end:${session.sessionId}`);
		});

		// Fire two focus requests without awaiting between them. Unserialized, the two
		// rebinds would interleave (start:b1, start:b2, end:b1, end:b2) and the final
		// rendered/subscribed state could point at the wrong session.
		const p1 = runtimeHost.focusSession(b1);
		const p2 = runtimeHost.focusSession(b2);
		await Promise.all([p1, p2]);

		expect(order).toEqual([`start:${b1}`, `end:${b1}`, `start:${b2}`, `end:${b2}`]);
		expect(runtimeHost.session.sessionId).toBe(b2);
	});

	it("emits session_blur on the outgoing session and session_focus on the newly focused one", async () => {
		let blurs = 0;
		let focuses = 0;
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_blur", () => {
				blurs++;
			});
			pi.on("session_focus", () => {
				focuses++;
			});
		});
		const second = await runtimeHost.createSession({ focus: false });

		// Focusing another session blurs the outgoing one (so its extensions tear down UI
		// before being backgrounded) and focuses the new one — both edges emitted by the
		// runtime after the rebind, in one layer, so extensions can reinstall focus-local UI.
		await runtimeHost.focusSession(second);
		expect(blurs).toBe(1);
		expect(focuses).toBe(1);
	});

	it("does not hang closing a backgrounded session whose shutdown handler prompts", async () => {
		let confirmResult: boolean | undefined;
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_shutdown", async (_event, ctx) => {
				confirmResult = await ctx.ui.confirm("Quit?", "Really?");
			});
		});
		// Mimic the InteractiveMode deferring background UI: a blurred session gets a UI
		// whose prompts never resolve (they would defer until a focus that never comes for
		// a closing session). closeSession must force a non-deferring UI before its
		// shutdown handlers run.
		runtimeHost.setBeforeSessionInvalidate(() => {
			const hangingUI = {
				confirm: () => new Promise<boolean>(() => {}),
			} as unknown as ExtensionUIContext;
			runtimeHost.session.detachUI(hangingUI);
		});

		const second = await runtimeHost.createSession({ focus: false });
		// Closing the focused session blurs it (installing the hanging UI above), then runs
		// its shutdown handler. The close must complete with confirm resolving to the no-op
		// default rather than hanging on the deferring UI.
		await runtimeHost.closeSession(runtimeHost.session.sessionId);
		expect(confirmResult).toBe(false);
		expect(runtimeHost.session.sessionId).toBe(second);
	});

	it("uses the background UI factory for attended creates but not autonomous ones", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const factoryCalls: string[] = [];
		runtimeHost.setBackgroundUIContextFactory((id) => {
			factoryCalls.push(id);
			return {} as unknown as ExtensionUIContext;
		});

		const attended = await runtimeHost.createSession({ focus: false });
		const autonomous = await runtimeHost.createSession({ focus: false, autonomous: true });

		// Attended background sessions get the deferring UI from creation; autonomous ones
		// keep the no-op UI (their prompts proceed with a safe default).
		expect(factoryCalls).toContain(attended);
		expect(factoryCalls).not.toContain(autonomous);
	});

	it("tracks the autonomous flag per session", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const attended = await runtimeHost.createSession({ focus: false });
		const autonomous = await runtimeHost.createSession({ focus: false, autonomous: true });

		expect(runtimeHost.isAutonomous(attended)).toBe(false);
		expect(runtimeHost.isAutonomous(autonomous)).toBe(true);
		// The initial focused session (and any replacement) is attended.
		expect(runtimeHost.isAutonomous(runtimeHost.session.sessionId)).toBe(false);
	});

	it("detachUI keeps interactive mode when given a deferring background context", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const runner = runtimeHost.session.extensionRunner;
		await runtimeHost.session.bindExtensions({ uiContext: {} as unknown as ExtensionUIContext, mode: "tui" });

		// An attended session backgrounded with a deferring context stays tui/hasUI, so
		// its gated extensions still call ctx.ui.* and those prompts defer.
		runtimeHost.session.detachUI({} as unknown as ExtensionUIContext);
		expect(runner.hasUI()).toBe(true);
		expect(runner.createContext().mode).toBe("tui");

		// With no background context (autonomous) it falls back to the no-op UI in print.
		runtimeHost.session.detachUI();
		expect(runner.hasUI()).toBe(false);
		expect(runner.createContext().mode).toBe("print");
	});

	it("does not deadlock when a hook re-enters a session operation", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		const b1 = await runtimeHost.createSession({ focus: false });
		const b2 = await runtimeHost.createSession({ focus: false });

		let reentered = false;
		runtimeHost.setRebindSession(async (session) => {
			// Simulate a hook (e.g. a session_focus handler) calling a public session
			// method while the focus op that triggered it is still running. This must run
			// inline; queuing it behind the outer op (which awaits this rebind) deadlocks.
			if (session.sessionId === b1 && !reentered) {
				reentered = true;
				await runtimeHost.focusSession(b2);
			}
		});

		await runtimeHost.focusSession(b1);
		expect(reentered).toBe(true);
		expect(runtimeHost.session.sessionId).toBe(b2);
	});

	it("focuses an already-live session on createSession({ resume, focus }) without duplicating", async () => {
		const { runtimeHost } = await createRuntimeHost(() => {});
		await runtimeHost.session.prompt("hello"); // persist the first session's file
		const firstId = runtimeHost.session.sessionId;
		const firstFile = runtimeHost.session.sessionFile;
		expect(firstFile).toBeTruthy();

		// Focus a different session so the first is backgrounded but still live.
		const second = await runtimeHost.createSession({ focus: true });
		expect(second).not.toBe(firstId);
		expect(runtimeHost.listSessions().length).toBe(2);

		// Resuming the already-live first session focuses the live copy instead of
		// opening a duplicate (and skips the cwd validation that runs only for a new
		// resume — so this works even if that cwd was since removed).
		const resumedId = await runtimeHost.createSession({ resume: firstFile!, focus: true });
		expect(resumedId).toBe(firstId);
		expect(runtimeHost.listSessions().length).toBe(2);
		expect(runtimeHost.session.sessionId).toBe(firstId);
	});

	it("routes background-session extension errors to the background error sink", async () => {
		const errors: string[] = [];
		const { runtimeHost } = await createRuntimeHost((pi) => {
			pi.on("session_start", (event) => {
				if (event.reason === "new") {
					throw new Error("boom in background session_start");
				}
			});
		});
		runtimeHost.setOnBackgroundError((error) => {
			errors.push(error.error);
		});

		// A background session's throwing session_start would otherwise have no error
		// listener and be silently dropped; the sink must receive it.
		await runtimeHost.createSession({ focus: false });
		expect(errors.some((e) => e.includes("boom"))).toBe(true);
	});
});
