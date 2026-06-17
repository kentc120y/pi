import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { cleanupRuntimes, createRuntimeForTest } from "./multi-session-helpers.ts";

describe("AgentSessionRuntime multi-session", () => {
	afterEach(cleanupRuntimes);

	it("holds multiple live sessions and focuses without disposing the others", async () => {
		const { runtime, faux } = await createRuntimeForTest(() => {});
		const firstId = runtime.session.sessionId;

		const secondId = await runtime.createSession();
		await runtime.getSession(secondId)!.bindExtensions({});

		expect(runtime.listSessions().map((s) => s.id)).toContain(firstId);
		expect(runtime.listSessions().map((s) => s.id)).toContain(secondId);
		expect(runtime.session.sessionId).toBe(firstId); // create does not auto-focus

		await runtime.focusSession(secondId);
		expect(runtime.session.sessionId).toBe(secondId);

		// the first session is still alive: focus back and prompt it
		await runtime.focusSession(firstId);
		faux.setResponses([fauxAssistantMessage("still here")]);
		await runtime.session.prompt("ping");
		expect(runtime.session.messages.at(-1)?.role).toBe("assistant");
	});

	it("closeSession disposes one session and falls back to another when it was focused", async () => {
		const { runtime } = await createRuntimeForTest(() => {});
		const firstId = runtime.session.sessionId;
		const secondId = await runtime.createSession();
		await runtime.getSession(secondId)!.bindExtensions({});

		await runtime.focusSession(secondId);
		await runtime.closeSession(secondId);
		expect(runtime.session.sessionId).toBe(firstId);
		expect(runtime.getSession(secondId)).toBeUndefined();
	});

	it("focusing an unknown id rejects", async () => {
		const { runtime } = await createRuntimeForTest(() => {});
		await expect(runtime.focusSession("does-not-exist")).rejects.toThrow();
	});

	it("closing the last remaining session throws", async () => {
		const { runtime } = await createRuntimeForTest(() => {});
		// The runtime must always have a focused session; closing the only one is rejected so
		// the host is never left with nothing to render or prompt.
		await expect(runtime.closeSession(runtime.session.sessionId)).rejects.toThrow(
			"Cannot close the last remaining session",
		);
		expect(runtime.listSessions().length).toBe(1);
	});

	it("closing the focused session does not throw when a before-invalidate hook reads the session", async () => {
		const { runtime } = await createRuntimeForTest(() => {});
		const firstId = runtime.session.sessionId;
		const secondId = await runtime.createSession();
		await runtime.focusSession(secondId);

		// Wire hooks like InteractiveMode does: the before-invalidate hook reads the
		// (outgoing) focused session, which previously threw mid-close.
		runtime.setBeforeSessionInvalidate(() => {
			void runtime.session.sessionId;
		});
		runtime.setRebindSession(async () => {});

		await expect(runtime.closeSession(secondId)).resolves.toBeUndefined();
		expect(runtime.session.sessionId).toBe(firstId);
	});
});
