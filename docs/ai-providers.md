# Native AI providers and model routing

Craft Anything's Rust agent loop builds projects through direct APIs. An external coding agent
is optional. Gemini and xAI use their providers' OpenAI-compatible REST endpoints; OpenAI uses
the Responses API. Anthropic, OpenRouter and local OpenAI-compatible servers continue to work.
The same engine tools, Clean room policy, approvals, project file restrictions and process
supervision apply to all API providers.

## Configure a provider

In Settings → AI, the AI Control Center offers Gemini and xAI even before a key is found. Paste
a key or set `GEMINI_API_KEY` / `XAI_API_KEY`, run Detect, and Add the provider. Keys use the existing
OS keychain or private fallback store, separate from project files and model catalogs.

The CLI accepts `--kind gemini` and `--kind xai`, alongside the existing backend kinds:

```sh
craft-anything-cli ai add Gemini --kind gemini --key-env GEMINI_API_KEY
craft-anything-cli ai test Gemini
craft-anything-cli ai models Gemini
craft-anything-cli ai builder Gemini
craft-anything-cli chat --backend Gemini "Describe the available tools"
```

Use the discovered model dropdown to change the configured model. A base URL override remains
available through CLI/configuration, including for local test servers. Gemini's default discovery
uses the native paginated Models API for generation methods and context limits. Custom Gemini
addresses use their compatible `/models` endpoint.

## Discovery and recommendations

Discover models refreshes one backend's saved catalog. A failed refresh leaves its previous
catalog intact. Discovery preserves supported tool parameters, input modalities, context limits,
deprecation flags and token prices when the endpoint publishes them. OpenAI-compatible model lists
often expose only ids; absent information remains unknown. A zero published price does not prove
account free-tier eligibility or remaining quota. Latency and quality are optional measured
metadata, not guesses based on model names.

```sh
craft-anything-cli ai models Router
craft-anything-cli ai recommend --strategy LOWEST_COST --tools --context-tokens 32000
craft-anything-cli ai routing LOCAL_FIRST
```

Available strategies are `AUTO`, `FREE_TIER_FIRST`, `LOWEST_COST`, `FASTEST`, `BEST_QUALITY` and
`LOCAL_FIRST`. AUTO retains the configured role and backend order. Other strategies rank models
that meet the task's capability, context and estimated-cost constraints; known ranking metadata
comes before unknown metadata, with catalog order breaking ties. The default cost comparison
estimates 1,000 input and 1,000 output tokens; library/tool callers can supply their own estimates.
Requiring tools or vision excludes models whose support is unknown. With no qualifying model,
routing returns a useful error rather than claiming compatibility. Explicit role, project and
backend choices take priority. Automatic native builds require verified tool support.

Only loopback OpenAI-compatible servers are classified as local. Hosted compatible endpoints
have unknown costs until a price is supplied. Discovered prices are scoped by backend and model;
explicit `[ai.prices]` overrides win. Token use and estimated costs feed the existing project
usage records and journal. This is an estimate, not a provider invoice or a hard spending cap.

## Headless tools

| Tool | Behavior |
|---|---|
| `settings_ai_models_list` | Read persisted model catalogs |
| `settings_ai_models_discover` | Refresh and persist a configured provider's models |
| `settings_ai_model_recommend` | Recommend a model using a strategy and task requirements |
| `settings_ai_routing_set` | Save the automatic routing strategy |
| `settings_ai_model_set` | Change one API backend's model id |

These tools are available through the engine, CLI and MCP, including in Clean room mode. They
operate on AI configuration, without reading an original application's internals.

## Runtime and recovery

The existing native agent loop maintains conversation context, invokes engine tools, handles
approval requests, writes files and runs project commands through the Build toolset. It keeps
project transcripts and summaries, and the rebuild workflow checks artifacts and requests fixes.
Native builders now honor the configured API fallback chain, skipping unavailable adapters.

The shared gateway retries transient failures at most twice with exponential backoff, provided
no response delta has been emitted. Three transient failures open its conversation-local circuit
for 30 seconds. Cancellation interrupts backoff. A partial answer is never replayed automatically.
HTTP connect, first-byte and body timeouts remain enforced by the transport. SSE responses are
bounded in size. Keys are redacted from HTTP and streamed transport errors before journaling.

Transcript version 2 preserves Responses encrypted reasoning and Gemini signed function metadata
as opaque provider data. Version 1 transcripts remain readable. Responses sends `store: false`;
Craft Anything persists its own context locally. Tool calls run only after a terminal Responses
event with valid arguments; incomplete streams cannot execute their partial calls.

## Validation and remaining work

Regression coverage includes loopback mock servers for OpenAI Responses, Gemini and xAI tool turns,
discovery persistence and routing, plus unit tests for ranking, replay, cancellation, circuit
breaking, credential redaction and headless Clean room configuration tools. Full crate tests
could not run in the implementation environment: crates.io DNS resolution failed and required
test dependencies were not cached. The desktop release build subsequently succeeded offline
with a writable Cargo cache, and its version smoke check passed. The Codex non-Git-folder
regression also passed against the compiled production AI library. Eighteen unit tests passed in a temporary offline harness using
the actual provider, stream, replay, gateway and routing modules with HTTP/evidence host stubs
and schema derives omitted. These isolated tests do not replace the full engine/UI test suites.
No live provider engineering build has been claimed.

Account-specific quota/free-tier monitoring, periodic discovery, structured answer schemas,
native image inputs, a dedicated specialist scheduler, semantic project memory, execution spending
caps and a provider health dashboard remain tracked under R2.1 in the roadmap. The current
implementation extends the existing runtime and does not claim the entire proposal is complete.

Provider formats follow the official [Gemini compatibility guide](https://ai.google.dev/gemini-api/docs/openai),
[Gemini Models API](https://ai.google.dev/api/models), [xAI API reference](https://docs.x.ai/developers/rest-api-reference),
and [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling).
