import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, registerFauxProvider } from "@earendil-works/pi-ai";
import {
	type CreateAgentSessionRuntimeFactory,
	createAgentSessionFromServices,
	createAgentSessionRuntime,
	createAgentSessionServices,
} from "../../src/core/agent-session-runtime.ts";
import { AuthStorage } from "../../src/core/auth-storage.ts";
import { SessionManager } from "../../src/core/session-manager.ts";
import type { ExtensionAPI, ExtensionFactory } from "../../src/index.ts";

const cleanups: Array<() => Promise<void> | void> = [];

/** Runs and clears all registered runtime cleanups. Register via `afterEach`. */
export async function cleanupRuntimes(): Promise<void> {
	while (cleanups.length > 0) {
		await cleanups.pop()?.();
	}
}

/**
 * Builds a fully wired AgentSessionRuntime backed by a faux provider for suite tests.
 *
 * The runtime is bound with `sessionsHost: runtime` so `pi.sessions` resolves against
 * the shared runtime host. Cleanup is registered on a module-level list drained by
 * `cleanupRuntimes()`.
 */
export async function createRuntimeForTest(
	extensionFactory: ExtensionFactory,
	options?: { cwd?: string; bootstrapModel?: boolean; bootstrapThinkingLevel?: boolean },
) {
	const tempDir =
		options?.cwd ?? join(tmpdir(), `pi-runtime-suite-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	mkdirSync(tempDir, { recursive: true });

	const faux = registerFauxProvider({
		models: [
			{ id: "faux-1", reasoning: true },
			{ id: "faux-2", reasoning: false },
		],
	});
	faux.setResponses([fauxAssistantMessage("one"), fauxAssistantMessage("two"), fauxAssistantMessage("three")]);

	const authStorage = AuthStorage.inMemory();
	authStorage.setRuntimeApiKey(faux.getModel().provider, "faux-key");

	const runtimeOptions = {
		agentDir: tempDir,
		authStorage,
		model: options?.bootstrapModel === false ? undefined : faux.getModel(),
		thinkingLevel: options?.bootstrapThinkingLevel === false ? undefined : undefined,
		resourceLoaderOptions: {
			extensionFactories: [
				(pi: ExtensionAPI) => {
					pi.registerProvider(faux.getModel().provider, {
						baseUrl: faux.getModel().baseUrl,
						apiKey: "faux-key",
						api: faux.api,
						models: faux.models.map((registeredModel) => ({
							id: registeredModel.id,
							name: registeredModel.name,
							api: registeredModel.api,
							reasoning: registeredModel.reasoning,
							input: registeredModel.input,
							cost: registeredModel.cost,
							contextWindow: registeredModel.contextWindow,
							maxTokens: registeredModel.maxTokens,
						})),
					});
					extensionFactory(pi);
				},
			],
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
		const created = await createAgentSessionFromServices({
			services,
			sessionManager,
			sessionStartEvent,
			model: runtimeOptions.model,
			thinkingLevel: runtimeOptions.thinkingLevel,
		});
		return {
			...created,
			services,
			diagnostics: services.diagnostics,
		};
	};
	const runtime = await createAgentSessionRuntime(createRuntime, {
		cwd: tempDir,
		agentDir: tempDir,
		sessionManager: SessionManager.create(tempDir),
	});
	await runtime.session.bindExtensions({ sessionsHost: runtime });

	cleanups.push(async () => {
		await runtime.dispose();
		faux.unregister();
		if (existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	return { runtime, faux, tempDir };
}
