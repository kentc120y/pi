import { AsyncLocalStorage } from "node:async_hooks";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { resolvePath } from "../utils/paths.ts";
import type { AgentSession } from "./agent-session.ts";
import type { AgentSessionRuntimeDiagnostic, AgentSessionServices } from "./agent-session-services.ts";
import type {
	ExtensionErrorListener,
	ExtensionEventObserver,
	ExtensionUIContext,
	ProjectTrustContext,
	ReplacedSessionContext,
	RuntimeSessionInfo,
	SessionShutdownEvent,
	SessionStartEvent,
	SessionsHost,
} from "./extensions/index.ts";
import { emitSessionShutdownEvent } from "./extensions/runner.ts";
import type { CreateAgentSessionResult } from "./sdk.ts";
import { assertSessionCwdExists } from "./session-cwd.ts";
import { loadEntriesFromFile, type SessionHeader, SessionManager } from "./session-manager.ts";

/**
 * Result returned by runtime creation.
 *
 * The caller gets the created session, its cwd-bound services, and all
 * diagnostics collected during setup.
 */
export interface CreateAgentSessionRuntimeResult extends CreateAgentSessionResult {
	services: AgentSessionServices;
	diagnostics: AgentSessionRuntimeDiagnostic[];
}

/**
 * Creates a full runtime for a target cwd and session manager.
 *
 * The factory closes over process-global fixed inputs, recreates cwd-bound
 * services for the effective cwd, resolves session options against those
 * services, and finally creates the AgentSession.
 */
export type CreateAgentSessionRuntimeFactory = (options: {
	cwd: string;
	agentDir: string;
	sessionManager: SessionManager;
	sessionStartEvent?: SessionStartEvent;
	projectTrustContext?: ProjectTrustContext;
}) => Promise<CreateAgentSessionRuntimeResult>;

/**
 * Thrown when /import references a JSONL file path that does not exist.
 */
export class SessionImportFileNotFoundError extends Error {
	readonly filePath: string;

	constructor(filePath: string) {
		super(`File not found: ${filePath}`);
		this.name = "SessionImportFileNotFoundError";
		this.filePath = filePath;
	}
}

/** Read a session's id from its JSONL header without opening/migrating the file. */
function readSessionIdFromFile(filePath: string): string | undefined {
	const header = loadEntriesFromFile(filePath).find((entry) => entry.type === "session") as SessionHeader | undefined;
	return header?.id;
}

function extractUserMessageText(content: string | Array<{ type: string; text?: string }>): string {
	if (typeof content === "string") {
		return content;
	}

	return content
		.filter((part): part is { type: "text"; text: string } => part.type === "text" && typeof part.text === "string")
		.map((part) => part.text)
		.join("");
}

interface RuntimeSessionEntry {
	session: AgentSession;
	services: AgentSessionServices;
	diagnostics: AgentSessionRuntimeDiagnostic[];
	modelFallbackMessage?: string;
	/**
	 * An autonomous (non-attended) session proceeds extension UI prompts with a safe
	 * default while backgrounded, instead of deferring them until focus. Only
	 * pi.sessions.create({ autonomous: true }) sets this; all others are attended.
	 */
	autonomous: boolean;
}

/**
 * Owns the current AgentSession plus its cwd-bound services.
 *
 * Session replacement methods tear down the current runtime first, then create
 * and apply the next runtime. If creation fails, the error is propagated to the
 * caller. The caller is responsible for user-facing error handling.
 */
export class AgentSessionRuntime implements SessionsHost {
	private rebindSession?: (session: AgentSession) => Promise<void>;
	private beforeSessionInvalidate?: () => void;
	private onSessionClosed?: (id: string) => void;
	private readonly createRuntime: CreateAgentSessionRuntimeFactory;
	private readonly _sessions = new Map<string, RuntimeSessionEntry>();
	private _focusedId: string;
	/** Tail of the focus-operation queue; see serializeFocusOp. */
	private _focusOpQueue: Promise<void> = Promise.resolve();
	/** Set while a serialized op (and its hooks) run, to detect re-entrant calls. */
	private readonly _opContext = new AsyncLocalStorage<true>();
	private projectTrustContextFactory?: (cwd: string) => ProjectTrustContext;
	private onBackgroundError?: ExtensionErrorListener;
	private backgroundUIContextFactory?: (sessionId: string) => ExtensionUIContext | undefined;
	/**
	 * Cross-session event observers (SessionHandle.on) keyed by observed session id, each
	 * mapped to its current detach fn. Keyed by id rather than bound to the AgentSession
	 * object so a same-id replacement can re-home them to the new object's runner. See
	 * observeSession.
	 */
	private readonly _sessionObservers = new Map<string, Map<ExtensionEventObserver, () => void>>();

	constructor(
		session: AgentSession,
		services: AgentSessionServices,
		createRuntime: CreateAgentSessionRuntimeFactory,
		diagnostics: AgentSessionRuntimeDiagnostic[] = [],
		modelFallbackMessage?: string,
	) {
		this.createRuntime = createRuntime;
		this._focusedId = session.sessionId;
		this._sessions.set(this._focusedId, { session, services, diagnostics, modelFallbackMessage, autonomous: false });
	}

	private get focusedEntry(): RuntimeSessionEntry {
		const entry = this._sessions.get(this._focusedId);
		if (!entry) {
			throw new Error("AgentSessionRuntime has no focused session");
		}
		return entry;
	}

	get services(): AgentSessionServices {
		return this.focusedEntry.services;
	}

	get session(): AgentSession {
		return this.focusedEntry.session;
	}

	get cwd(): string {
		return this.focusedEntry.services.cwd;
	}

	get diagnostics(): readonly AgentSessionRuntimeDiagnostic[] {
		return this.focusedEntry.diagnostics;
	}

	get modelFallbackMessage(): string | undefined {
		return this.focusedEntry.modelFallbackMessage;
	}

	setRebindSession(rebindSession?: (session: AgentSession) => Promise<void>): void {
		this.rebindSession = rebindSession;
	}

	/**
	 * Set a synchronous callback that runs after `session_shutdown` handlers finish
	 * but before the current session is invalidated.
	 *
	 * This is for host-owned UI teardown that must not yield to the event loop,
	 * such as detaching extension-provided TUI components before the old extension
	 * context becomes stale.
	 */
	setBeforeSessionInvalidate(beforeSessionInvalidate?: () => void): void {
		this.beforeSessionInvalidate = beforeSessionInvalidate;
	}

	/**
	 * Set a callback that runs after a session has been closed and removed from the
	 * runtime, with the closed session's id. Lets the host drop any per-session
	 * state it keeps keyed by session id (e.g. retained TUI state) so it does not
	 * leak for the lifetime of the process.
	 */
	setOnSessionClosed(onSessionClosed?: (id: string) => void): void {
		this.onSessionClosed = onSessionClosed;
	}

	/**
	 * Set the factory used to resolve project trust for a focused, API-created resume
	 * (pi.sessions.create({ resume, focus: true })) — the same interactive trust
	 * context /resume uses, so an extension-triggered focused resume prompts instead
	 * of silently running untrusted. Background (unfocused) creates never prompt.
	 */
	setProjectTrustContextFactory(factory?: (cwd: string) => ProjectTrustContext): void {
		this.projectTrustContextFactory = factory;
	}

	/**
	 * Set a sink for extension errors raised by background (unfocused) sessions, which
	 * otherwise have no error listener bound — so a throwing session_start, resource
	 * discovery, or event handler on a background session would be silently dropped.
	 */
	setOnBackgroundError(onBackgroundError?: ExtensionErrorListener): void {
		this.onBackgroundError = onBackgroundError;
	}

	/**
	 * Set the factory that builds the deferring UI context for a background, attended
	 * session (so its prompts during session_start or a background turn wait for focus
	 * instead of resolving with autonomous defaults). Returns undefined for headless
	 * hosts. Autonomous sessions never use it.
	 */
	setBackgroundUIContextFactory(factory?: (sessionId: string) => ExtensionUIContext | undefined): void {
		this.backgroundUIContextFactory = factory;
	}

	/**
	 * Serialize a focus-changing operation. focusSession/close/create/switch/new/fork/
	 * import all mutate _focusedId/_sessions and await a rebind; without serialization
	 * two concurrent calls (e.g. two extension handle.focus() calls, or a focus during
	 * a close) could interleave their awaits and leave focus, render, subscription, or
	 * draft state pointing at the wrong final target. Each op waits for the previous to
	 * settle, then runs. The tail advances on both success and failure so one rejection
	 * cannot wedge the queue.
	 *
	 * A serialized op awaits hooks (extension event handlers, withSession callbacks, the
	 * rebind) that can themselves call a public session method. Such a re-entrant call
	 * MUST run inline — queuing it behind the op that is awaiting its hook would make
	 * both wait forever. We detect re-entrancy with an AsyncLocalStorage flag that is
	 * set across the op and all its awaited continuations; only top-level calls (no flag,
	 * i.e. from the event loop) queue.
	 */
	private serializeFocusOp<T>(op: () => Promise<T>): Promise<T> {
		if (this._opContext.getStore()) {
			return op();
		}
		const result = this._focusOpQueue.then(() => this._opContext.run(true, op));
		this._focusOpQueue = result.then(
			() => {},
			() => {},
		);
		return result;
	}

	private async emitBeforeSwitch(
		reason: "new" | "resume",
		targetSessionFile?: string,
	): Promise<{ cancelled: boolean }> {
		const runner = this.session.extensionRunner;
		// Also emit when a cross-session observer is watching, so a
		// handle.on("session_before_switch", ...) observer fires even with no own handler.
		if (!runner.hasHandlers("session_before_switch") && !runner.hasObservers()) {
			return { cancelled: false };
		}

		const result = await runner.emit({
			type: "session_before_switch",
			reason,
			targetSessionFile,
		});
		return { cancelled: result?.cancel === true };
	}

	private async emitBeforeFork(
		entryId: string,
		options: { position: "before" | "at" },
	): Promise<{ cancelled: boolean }> {
		const runner = this.session.extensionRunner;
		if (!runner.hasHandlers("session_before_fork") && !runner.hasObservers()) {
			return { cancelled: false };
		}

		const result = await runner.emit({
			type: "session_before_fork",
			entryId,
			...options,
		});
		return { cancelled: result?.cancel === true };
	}

	private async teardownCurrent(reason: SessionShutdownEvent["reason"], targetSessionFile?: string): Promise<void> {
		await emitSessionShutdownEvent(this.session.extensionRunner, {
			type: "session_shutdown",
			reason,
			targetSessionFile,
		});
		this.beforeSessionInvalidate?.();
		this.session.dispose();
	}

	private apply(result: CreateAgentSessionRuntimeResult): void {
		// The outgoing focused session OBJECT was just disposed by teardownCurrent and is now
		// removed for good; notify so the host drops any transient state keyed to it (retained
		// UI state, history-populated flag, deferred prompts). Fire even when the replacement
		// reuses the same id — /resume or /import of the focused session disposes and recreates
		// it under the same id, and that stale per-session state must not carry into the fresh
		// session (which repopulates its own on rebind). The host handler only clears id-keyed
		// transient maps, so a same-id refire is safe.
		const replacedId = this._focusedId;
		this._sessions.delete(replacedId);
		this._focusedId = result.session.sessionId;
		this.onSessionClosed?.(replacedId);
		this._sessions.set(this._focusedId, {
			session: result.session,
			services: result.services,
			diagnostics: result.diagnostics,
			modelFallbackMessage: result.modelFallbackMessage,
			// A replacement (new/fork/resume/import) is the focused, user-driven session.
			autonomous: false,
		});
		if (replacedId === this._focusedId) {
			// Same-id replacement (/resume or /import of the focused session): the old object was
			// disposed and a new one installed under the same id. Re-home cross-session observers
			// (SessionHandle.on) to the new runner so they keep delivering.
			this.rebindSessionObservers(this._focusedId, result.session);
		} else {
			// The replaced id is gone for good; drop any observers that were watching it.
			this.dropSessionObservers(replacedId);
		}
	}

	private async finishSessionReplacement(withSession?: (ctx: ReplacedSessionContext) => Promise<void>): Promise<void> {
		const focusedId = this._focusedId;
		if (this.rebindSession) {
			await this.rebindSession(this.session);
		}
		// A replacement (new/resume/fork/import) installs a freshly focused session; emit
		// session_focus from the runtime after the rebind, like _focusSession, so the focus edge
		// is owned in one layer. Before withSession, matching the prior ordering when the TUI
		// rebind emitted it. Guard against a re-entrant focus change during the rebind (see
		// _focusSession).
		if (this._focusedId === focusedId) {
			await this.session.emitSessionFocus();
		}
		if (withSession) {
			await withSession(this.session.createReplacedSessionContext());
		}
	}

	async switchSession(
		sessionPath: string,
		options?: { cwdOverride?: string; withSession?: (ctx: ReplacedSessionContext) => Promise<void> },
	): Promise<{ cancelled: boolean }> {
		return this.serializeFocusOp(() => this._switchSession(sessionPath, options));
	}

	private async _switchSession(
		sessionPath: string,
		options?: {
			cwdOverride?: string;
			withSession?: (ctx: ReplacedSessionContext) => Promise<void>;
			projectTrustContextFactory?: (cwd: string) => ProjectTrustContext;
		},
	): Promise<{ cancelled: boolean }> {
		const beforeResult = await this.emitBeforeSwitch("resume", sessionPath);
		if (beforeResult.cancelled) {
			return beforeResult;
		}

		const previousSessionFile = this.session.sessionFile;
		const sessionManager = SessionManager.open(sessionPath, undefined, options?.cwdOverride);

		// If the resumed session is already live in the background, focus that copy
		// instead of opening a second live one that would double-write the same file.
		// This runs BEFORE assertSessionCwdExists: focusing an already-live session
		// must not re-validate its cwd (which would throw if the cwd was since removed)
		// — the live copy is already running fine. focusSession() re-binds the target's
		// UI; still run withSession afterwards so the post-switch contract holds.
		const targetId = sessionManager.getSessionId();
		if (targetId !== this._focusedId && this._sessions.has(targetId)) {
			await this._focusSession(targetId);
			if (options?.withSession) {
				await options.withSession(this.session.createReplacedSessionContext());
			}
			return { cancelled: false };
		}

		assertSessionCwdExists(sessionManager, this.cwd);

		await this.teardownCurrent("resume", sessionManager.getSessionFile());
		this.apply(
			await this.createRuntime({
				cwd: sessionManager.getCwd(),
				agentDir: this.services.agentDir,
				sessionManager,
				sessionStartEvent: { type: "session_start", reason: "resume", previousSessionFile },
			}),
		);
		await this.finishSessionReplacement(options?.withSession);
		return { cancelled: false };
	}

	async newSession(options?: {
		parentSession?: string;
		setup?: (sessionManager: SessionManager) => Promise<void>;
		withSession?: (ctx: ReplacedSessionContext) => Promise<void>;
	}): Promise<{ cancelled: boolean }> {
		return this.serializeFocusOp(() => this._newSession(options));
	}

	private async _newSession(options?: {
		parentSession?: string;
		setup?: (sessionManager: SessionManager) => Promise<void>;
		withSession?: (ctx: ReplacedSessionContext) => Promise<void>;
	}): Promise<{ cancelled: boolean }> {
		const beforeResult = await this.emitBeforeSwitch("new");
		if (beforeResult.cancelled) {
			return beforeResult;
		}

		const previousSessionFile = this.session.sessionFile;
		const sessionDir = this.session.sessionManager.getSessionDir();
		const sessionManager = SessionManager.create(this.cwd, sessionDir);
		if (options?.parentSession) {
			sessionManager.newSession({ parentSession: options.parentSession });
		}

		await this.teardownCurrent("new", sessionManager.getSessionFile());
		this.apply(
			await this.createRuntime({
				cwd: this.cwd,
				agentDir: this.services.agentDir,
				sessionManager,
				sessionStartEvent: { type: "session_start", reason: "new", previousSessionFile },
			}),
		);
		if (options?.setup) {
			await options.setup(this.session.sessionManager);
			this.session.agent.state.messages = this.session.sessionManager.buildSessionContext().messages;
		}
		await this.finishSessionReplacement(options?.withSession);
		return { cancelled: false };
	}

	async fork(
		entryId: string,
		options?: { position?: "before" | "at"; withSession?: (ctx: ReplacedSessionContext) => Promise<void> },
	): Promise<{ cancelled: boolean; selectedText?: string }> {
		return this.serializeFocusOp(() => this._fork(entryId, options));
	}

	private async _fork(
		entryId: string,
		options?: { position?: "before" | "at"; withSession?: (ctx: ReplacedSessionContext) => Promise<void> },
	): Promise<{ cancelled: boolean; selectedText?: string }> {
		const position = options?.position ?? "before";
		const beforeResult = await this.emitBeforeFork(entryId, { position });
		if (beforeResult.cancelled) {
			return { cancelled: true };
		}
		let targetLeafId: string | null;
		let selectedText: string | undefined;

		const selectedEntry = this.session.sessionManager.getEntry(entryId);
		if (!selectedEntry) {
			throw new Error("Invalid entry ID for forking");
		}

		if (position === "at") {
			targetLeafId = selectedEntry.id;
		} else {
			if (selectedEntry.type !== "message" || selectedEntry.message.role !== "user") {
				throw new Error("Invalid entry ID for forking");
			}
			targetLeafId = selectedEntry.parentId;
			selectedText = extractUserMessageText(selectedEntry.message.content);
		}

		const previousSessionFile = this.session.sessionFile;
		if (this.session.sessionManager.isPersisted()) {
			const currentSessionFile = this.session.sessionFile;
			if (!currentSessionFile) {
				throw new Error("Persisted session is missing a session file");
			}
			const sessionDir = this.session.sessionManager.getSessionDir();
			if (!targetLeafId) {
				const sessionManager = SessionManager.create(this.cwd, sessionDir);
				sessionManager.newSession({ parentSession: currentSessionFile });
				await this.teardownCurrent("fork", sessionManager.getSessionFile());
				this.apply(
					await this.createRuntime({
						cwd: this.cwd,
						agentDir: this.services.agentDir,
						sessionManager,
						sessionStartEvent: { type: "session_start", reason: "fork", previousSessionFile },
					}),
				);
				await this.finishSessionReplacement(options?.withSession);
				return { cancelled: false, selectedText };
			}

			const sessionManager = SessionManager.open(currentSessionFile, sessionDir);
			const forkedSessionPath = sessionManager.createBranchedSession(targetLeafId);
			if (!forkedSessionPath) {
				throw new Error("Failed to create forked session");
			}
			await this.teardownCurrent("fork", sessionManager.getSessionFile());
			this.apply(
				await this.createRuntime({
					cwd: sessionManager.getCwd(),
					agentDir: this.services.agentDir,
					sessionManager,
					sessionStartEvent: { type: "session_start", reason: "fork", previousSessionFile },
				}),
			);
			await this.finishSessionReplacement(options?.withSession);
			return { cancelled: false, selectedText };
		}

		const sessionManager = this.session.sessionManager;
		if (!targetLeafId) {
			sessionManager.newSession({ parentSession: this.session.sessionFile });
		} else {
			sessionManager.createBranchedSession(targetLeafId);
		}
		await this.teardownCurrent("fork", sessionManager.getSessionFile());
		this.apply(
			await this.createRuntime({
				cwd: this.cwd,
				agentDir: this.services.agentDir,
				sessionManager,
				sessionStartEvent: { type: "session_start", reason: "fork", previousSessionFile },
			}),
		);
		await this.finishSessionReplacement(options?.withSession);
		return { cancelled: false, selectedText };
	}

	/**
	 * Import a session JSONL file and switch runtime state to the imported session.
	 *
	 * @returns `{ cancelled: true }` when cancelled by `session_before_switch`, otherwise `{ cancelled: false }`.
	 * @throws {SessionImportFileNotFoundError} When the input path does not exist.
	 * @throws {MissingSessionCwdError} When the imported session cwd cannot be resolved and no override is provided.
	 */
	async importFromJsonl(inputPath: string, cwdOverride?: string): Promise<{ cancelled: boolean }> {
		return this.serializeFocusOp(() => this._importFromJsonl(inputPath, cwdOverride));
	}

	private async _importFromJsonl(inputPath: string, cwdOverride?: string): Promise<{ cancelled: boolean }> {
		const resolvedPath = resolvePath(inputPath);
		if (!existsSync(resolvedPath)) {
			throw new SessionImportFileNotFoundError(resolvedPath);
		}

		const sessionDir = this.session.sessionManager.getSessionDir();
		if (!existsSync(sessionDir)) {
			mkdirSync(sessionDir, { recursive: true });
		}

		const destinationPath = join(sessionDir, basename(resolvedPath));
		const beforeResult = await this.emitBeforeSwitch("resume", destinationPath);
		if (beforeResult.cancelled) {
			return beforeResult;
		}

		// If the imported session is already live in the background, focus that copy
		// and skip the copy entirely. This must happen BEFORE copyFileSync: the copy
		// would otherwise clobber the live session's own file (when basenames collide)
		// while it is still appending, corrupting it. Peek the id from the source
		// read-only so we neither open a second live copy nor migrate the source file.
		const importedId = readSessionIdFromFile(resolvedPath);
		if (importedId && importedId !== this._focusedId && this._sessions.has(importedId)) {
			await this._focusSession(importedId);
			return { cancelled: false };
		}

		const previousSessionFile = this.session.sessionFile;
		if (resolve(destinationPath) !== resolvedPath) {
			copyFileSync(resolvedPath, destinationPath);
		}

		const sessionManager = SessionManager.open(destinationPath, sessionDir, cwdOverride);
		assertSessionCwdExists(sessionManager, this.cwd);

		await this.teardownCurrent("resume", sessionManager.getSessionFile());
		this.apply(
			await this.createRuntime({
				cwd: sessionManager.getCwd(),
				agentDir: this.services.agentDir,
				sessionManager,
				sessionStartEvent: { type: "session_start", reason: "resume", previousSessionFile },
			}),
		);
		await this.finishSessionReplacement();
		return { cancelled: false };
	}

	async dispose(): Promise<void> {
		// Let any in-flight focus operation settle before tearing everything down, so
		// dispose never races a focus change mid-rebind.
		await this._focusOpQueue.catch(() => {});
		for (const entry of this._sessions.values()) {
			// Drop any deferring background UI first (same reason as closeSession): a
			// backgrounded attended session's shutdown handler must not wait on a focus
			// that will never happen while we await it.
			entry.session.detachUI();
			await emitSessionShutdownEvent(entry.session.extensionRunner, {
				type: "session_shutdown",
				reason: "quit",
			});
		}
		this.beforeSessionInvalidate?.();
		for (const entry of this._sessions.values()) {
			entry.session.dispose();
		}
		this._sessions.clear();
	}

	listSessions(): RuntimeSessionInfo[] {
		return [...this._sessions.entries()].map(([id, entry]) => ({
			id,
			name: entry.session.sessionName,
			isFocused: id === this._focusedId,
			isBusy:
				entry.session.isStreaming ||
				entry.session.isCompacting ||
				entry.session.isRetrying ||
				entry.session.isBashRunning,
		}));
	}

	getSession(id: string): AgentSession | undefined {
		return this._sessions.get(id)?.session;
	}

	/**
	 * Observe a session's mapped extension events by id, surviving a same-id replacement.
	 * SessionHandle.on() uses this instead of binding to the AgentSession object directly:
	 * /resume or /import of the focused session disposes the old object and installs a new
	 * one under the same id (see apply()), and an observer bound to the old object would
	 * silently stop. Keyed by id, the observer is re-homed to the new object's runner on
	 * replacement (rebindSessionObservers) and dropped when the id closes for good
	 * (dropSessionObservers). Returns an unsubscribe fn; a no-op if the id is not live.
	 */
	observeSession(id: string, observer: ExtensionEventObserver): () => void {
		const session = this._sessions.get(id)?.session;
		if (!session) {
			return () => {};
		}
		let observers = this._sessionObservers.get(id);
		if (!observers) {
			observers = new Map();
			this._sessionObservers.set(id, observers);
		}
		observers.set(observer, session.observeExtensionEvents(observer));
		return () => {
			const current = this._sessionObservers.get(id);
			if (!current) {
				return;
			}
			current.get(observer)?.();
			current.delete(observer);
			if (current.size === 0) {
				this._sessionObservers.delete(id);
			}
		};
	}

	/**
	 * Re-attach an id's cross-session observers to a freshly installed session object after a
	 * same-id replacement. The previous runner was just disposed, so its stored detach is moot;
	 * attach to the new runner and remember the new detach so unsubscribe still works.
	 */
	private rebindSessionObservers(id: string, session: AgentSession): void {
		const observers = this._sessionObservers.get(id);
		if (!observers) {
			return;
		}
		for (const observer of [...observers.keys()]) {
			observers.set(observer, session.observeExtensionEvents(observer));
		}
	}

	/** Detach and forget an id's cross-session observers once the id is gone for good. */
	private dropSessionObservers(id: string): void {
		const observers = this._sessionObservers.get(id);
		if (!observers) {
			return;
		}
		for (const detach of observers.values()) {
			detach();
		}
		this._sessionObservers.delete(id);
	}

	async focusSession(id: string): Promise<void> {
		return this.serializeFocusOp(() => this._focusSession(id));
	}

	/**
	 * Unlocked focus core. Callers already inside a serialized op (close/create/switch/
	 * import) use this directly; the public focusSession wraps it in serializeFocusOp.
	 */
	private async _focusSession(id: string): Promise<void> {
		if (!this._sessions.has(id)) {
			throw new Error(`Unknown session: ${id}`);
		}
		if (id === this._focusedId) {
			return;
		}
		const outgoingId = this._focusedId;
		// Tell the outgoing session's extensions it is being backgrounded while it still
		// has its UI — before beforeSessionInvalidate() detaches it.
		await this.session.emitSessionBlur();
		// A session_blur handler may re-enter pi.sessions.focus()/close() — detected as
		// re-entrant and run inline, so it completes (invalidate + rebind) before we resume
		// here. If it moved focus, finishing this outer focus would overwrite the focused id
		// it set and rebind a second time. Abort and let the nested op's result stand.
		if (this._focusedId !== outgoingId) {
			return;
		}
		this.beforeSessionInvalidate?.();
		this._focusedId = id;
		if (this.rebindSession) {
			await this.rebindSession(this.session);
		}
		// Tell the now-focused session's extensions it is focused, after the host's rebind
		// (re)installed its UI — pairs with the emitSessionBlur() above so both focus edges are
		// emitted from the runtime, in one layer. Guard against re-entrancy: a hook invoked
		// during the rebind may have re-entrantly moved focus elsewhere (that nested op already
		// emitted session_focus for the new target); don't re-emit for a session that is no
		// longer focused.
		if (this._focusedId === id) {
			await this.session.emitSessionFocus();
		}
	}

	async createSession(options?: { resume?: string; focus?: boolean; autonomous?: boolean }): Promise<string> {
		return this.serializeFocusOp(() => this._createSession(options));
	}

	/** Whether a session proceeds backgrounded prompts autonomously (vs deferring them). */
	isAutonomous(id: string): boolean {
		return this._sessions.get(id)?.autonomous ?? false;
	}

	private async _createSession(options?: { resume?: string; focus?: boolean; autonomous?: boolean }): Promise<string> {
		let sessionManager: SessionManager;
		let cwd: string;
		if (options?.resume) {
			// A resumed session runs in its own stored cwd (mirroring /resume via
			// switchSession). Cwd existence is validated below, AFTER the focus-existing
			// check, so re-resuming an already-live session whose cwd was since removed
			// just focuses the live copy instead of throwing.
			sessionManager = SessionManager.open(options.resume, undefined);
			cwd = sessionManager.getCwd();
		} else {
			// New sessions always run in the focused session's cwd. Creating a session
			// in a different cwd is intentionally unsupported (it has no interactive
			// project-trust path and would persist under the wrong session directory).
			cwd = this.cwd;
			sessionManager = this.session.sessionManager.isPersisted()
				? SessionManager.create(cwd, this.session.sessionManager.getSessionDir())
				: SessionManager.inMemory(cwd);
		}

		const existingId = sessionManager.getSessionId();
		if (this._sessions.has(existingId)) {
			if (options?.focus) {
				await this._focusSession(existingId);
			}
			return existingId;
		}

		// Only a genuinely new (not already-live) resume validates its stored cwd; a
		// new session uses this.cwd, which exists. Matches switchSession's ordering.
		if (options?.resume) {
			assertSessionCwdExists(sessionManager, this.cwd);
		}

		const result = await this.createRuntime({
			cwd,
			agentDir: this.services.agentDir,
			sessionManager,
			sessionStartEvent: options?.resume
				? { type: "session_start", reason: "resume" }
				: { type: "session_start", reason: "new" },
			// A focused resume resolves project trust like /resume (interactive prompt)
			// rather than silently running untrusted. Background creates pass no context
			// (no UI to prompt on) and stay untrusted; the trust warning shows if/when
			// they are later focused.
			projectTrustContext: options?.resume && options?.focus ? this.projectTrustContextFactory?.(cwd) : undefined,
		});
		const entry: RuntimeSessionEntry = {
			session: result.session,
			services: result.services,
			diagnostics: result.diagnostics,
			modelFallbackMessage: result.modelFallbackMessage,
			autonomous: options?.autonomous ?? false,
		};
		this._sessions.set(result.session.sessionId, entry);
		// session_start (and resource discovery) fire once, on a session's first bindExtensions,
		// in whatever mode that bind carries. A focused TUI create must fire them in TUI mode so
		// extensions that build their TUI UI on session_start initialize for it; the once-only
		// gate would suppress session_start there if we pre-bound in print mode first. So a
		// focused TUI create lets focusSession() below do the first bind. Everything else (a
		// background create, or a headless focused create with no TUI rebind) pre-binds here.
		const willFocusInTui = Boolean(options?.focus && this.rebindSession);
		if (!willFocusInTui) {
			// Bind a background session NON-INTERACTIVELY (no-op UI, print mode) and AWAIT it,
			// so session_start + resource discovery complete before createSession resolves — the
			// session is never observable half-started, and handle.sendUserMessage needs no
			// readiness gate. A no-op UI can't block on a human, so awaiting under the focus lock
			// can't deadlock (unlike a deferring UI, whose drain would need this very lock).
			//
			// LIMITATION: a session_start (or resource-discovery) handler that prompts via
			// ctx.ui.* on a background create gets the dialog's safe default (it runs in print
			// mode, ctx.hasUI false) and is NOT re-asked on focus. Extensions that must ask at
			// setup should detect ctx.mode/!ctx.hasUI and prompt from a session_focus handler.
			// Turn-time prompts still block & surface on focus — see the deferring UI armed below.
			// Background binds get the runtime's background error sink so a throwing handler is
			// not silently dropped.
			await result.session.bindExtensions({
				sessionsHost: this,
				onError: this.onBackgroundError,
				mode: "print",
			});
			// After startup, arm an ATTENDED background session with the deferring UI so its
			// TURN prompts (e.g. via handle.sendUserMessage) block and replay when the session is
			// focused, instead of resolving with autonomous defaults — exactly what blurring an
			// attended session installs. Autonomous sessions (and headless hosts, where the
			// factory is unset) keep the no-op UI and proceed with safe defaults.
			if (!options?.autonomous) {
				const deferringUI = this.backgroundUIContextFactory?.(result.session.sessionId);
				if (deferringUI) {
					result.session.detachUI(deferringUI);
				}
			}
		}
		if (options?.focus) {
			await this._focusSession(result.session.sessionId);
		}
		return result.session.sessionId;
	}

	async closeSession(id: string): Promise<void> {
		return this.serializeFocusOp(() => this._closeSession(id));
	}

	private async _closeSession(id: string): Promise<void> {
		const entry = this._sessions.get(id);
		if (!entry) {
			return;
		}
		if (this._sessions.size === 1) {
			throw new Error("Cannot close the last remaining session");
		}
		// If closing the focused session, focus another live session first so the
		// host UI detaches while the outgoing session is still resolvable; only
		// then tear the closed session down as a (now background) session.
		if (this._focusedId === id) {
			const next = [...this._sessions.keys()].find((key) => key !== id);
			if (!next) {
				throw new Error("closeSession: expected a remaining session");
			}
			await this._focusSession(next);
		}
		// Force the closing session's extensions onto the no-op UI before its shutdown
		// handlers run. If it kept the deferring background UI (an attended session just
		// blurred above), a session_shutdown handler calling ctx.ui.* would block on a
		// focus that never comes — and we are awaiting that handler — hanging the close.
		entry.session.detachUI();
		await emitSessionShutdownEvent(entry.session.extensionRunner, {
			type: "session_shutdown",
			reason: "quit",
		});
		entry.session.dispose();
		this._sessions.delete(id);
		this.dropSessionObservers(id);
		this.onSessionClosed?.(id);
	}
}

/**
 * Create the initial runtime from a runtime factory and initial session target.
 *
 * The same factory is stored on the returned AgentSessionRuntime and reused for
 * later /new, /resume, /fork, and import flows.
 */
export async function createAgentSessionRuntime(
	createRuntime: CreateAgentSessionRuntimeFactory,
	options: {
		cwd: string;
		agentDir: string;
		sessionManager: SessionManager;
		sessionStartEvent?: SessionStartEvent;
	},
): Promise<AgentSessionRuntime> {
	assertSessionCwdExists(options.sessionManager, options.cwd);
	const result = await createRuntime(options);
	return new AgentSessionRuntime(
		result.session,
		result.services,
		createRuntime,
		result.diagnostics,
		result.modelFallbackMessage,
	);
}

export {
	type AgentSessionRuntimeDiagnostic,
	type AgentSessionServices,
	type CreateAgentSessionFromServicesOptions,
	type CreateAgentSessionServicesOptions,
	createAgentSessionFromServices,
	createAgentSessionServices,
} from "./agent-session-services.ts";
