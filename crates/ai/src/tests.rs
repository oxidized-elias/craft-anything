//! End-to-end tests: a tiny std TcpListener HTTP server stands in for the model APIs, and a real
//! engine runs the tools. No real network.

use super::*;
use craft_anything_platform::sync::Mutex;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

#[derive(Debug, Clone)]
struct Seen {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Seen {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
}

/// Serve canned responses `(status, content_type, body)` in order; record requests.
fn mock(responses: Vec<(u16, &'static str, String)>) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for (status, ctype, reply) in responses {
            let Ok((stream, _)) = l.accept() else { return };
            let mut r = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or_default().to_string();
            let mut headers = Vec::new();
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                let h = h.trim_end().to_string();
                if h.is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap();
                    }
                    headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            let mut buf = vec![0; len];
            r.read_exact(&mut buf).unwrap();
            let body = serde_json::from_slice(&buf).unwrap_or(Value::Null);
            s2.lock().unwrap().push(Seen { path, headers, body });
            let mut w = stream;
            let reason = if status == 200 { "OK" } else { "Error" };
            write!(w, "HTTP/1.1 {status} {reason}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}", reply.len()).unwrap();
            w.flush().unwrap();
        }
    });
    (url, seen)
}

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("event: {}\r\ndata: {}\r\n\r\n", e["type"].as_str().unwrap_or("message"), e)).collect()
}

fn chat_sse(chunks: &[Value]) -> String {
    let mut s: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
    s.push_str("data: [DONE]\n\n");
    s
}

fn hub() -> (AiHub, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(Engine::new(craft_anything_engine::testing::session()));
    let h = AiHub::with_stores(engine, dir.path().join("config.toml"), KeyStore::file_only(dir.path().join("ai-keys.toml")));
    (h, dir)
}

#[test]
fn a_failed_save_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    // The settings "file" can't be written: its folder is a file.
    std::fs::write(dir.path().join("blocked"), "").unwrap();
    let engine = Arc::new(Engine::new(craft_anything_engine::testing::session()));
    let h = AiHub::with_stores(engine, dir.path().join("blocked").join("config.toml"), KeyStore::file_only(dir.path().join("ai-keys.toml")));
    let cfg = BackendConfig { name: "Kimi".into(), kind: BackendKind::Openrouter, model: "m".into(), base_url: None, ssh_host: None, enabled: true };
    assert!(h.add_backend(cfg, None).is_err());
    assert!(h.backends().is_empty(), "the app shows only what was saved");
}

#[test]
fn the_person_chooses_who_builds() {
    let (h, _d) = hub();
    add(&h, "Claude Code", BackendKind::AgentCli, "http://127.0.0.1:9", None);
    add(&h, "Kimi K3 (OpenRouter)", BackendKind::Openrouter, "http://127.0.0.1:9", Some("sk-test"));
    assert_eq!(h.chosen(Role::Builder), None, "automatic until chosen");
    assert!(h.set_role(Role::Builder, Some("Nobody")).is_err(), "only a configured backend");
    h.set_role(Role::Builder, Some("Kimi K3 (OpenRouter)")).unwrap();
    assert_eq!(h.chosen(Role::Builder).as_deref(), Some("Kimi K3 (OpenRouter)"));
    // The extractor is chosen the same way, and has no default.
    assert_eq!(h.chosen(Role::Extractor), None);
    h.set_role(Role::Extractor, Some("Kimi K3 (OpenRouter)")).unwrap();
    assert_eq!(h.chosen(Role::Extractor).as_deref(), Some("Kimi K3 (OpenRouter)"));
    // A removed backend can't stay the builder.
    h.remove_backend("Kimi K3 (OpenRouter)").unwrap();
    assert_eq!((h.chosen(Role::Builder), h.chosen(Role::Extractor)), (None, None));
    h.set_role(Role::Builder, Some("Claude Code")).unwrap();
    h.set_role(Role::Builder, None).unwrap();
    assert_eq!(h.chosen(Role::Builder), None, "back to automatic");
}

fn add(h: &AiHub, name: &str, kind: BackendKind, base: &str, key: Option<&str>) {
    h.add_backend(
        BackendConfig { name: name.into(), kind, model: String::new(), base_url: Some(base.into()), ssh_host: None, enabled: true },
        key.map(str::to_string),
    )
    .unwrap();
}

fn run(c: &mut Box<dyn Conversation>, text: &str) -> Vec<AgentEvent> {
    c.send(text);
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut all = Vec::new();
    while Instant::now() < deadline {
        for e in c.poll() {
            let done = matches!(e, AgentEvent::TurnDone { .. });
            all.push(e);
            if done {
                return all;
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out: {all:?}");
}

#[test]
fn anthropic_streaming_tool_loop_through_the_engine() {
    let turn1 = sse(&[
        json!({"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":100,"output_tokens":1}}}),
        json!({"type":"ping"}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu_1","name":"core_session_show","input":{}}}),
        json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{"}}),
        json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"}"}}),
        json!({"type":"content_block_stop","index":2}),
        json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"tu_2","name":"core_targets_list","input":{}}}),
        json!({"type":"content_block_stop","index":3}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}),
        json!({"type":"message_stop"}),
    ]);
    let turn2 = sse(&[
        json!({"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":200,"cache_read_input_tokens":90}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Nothing is open."}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}),
        json!({"type":"message_stop"}),
    ]);
    let (url, seen) = mock(vec![(200, "text/event-stream", turn1), (200, "text/event-stream", turn2)]);
    let (h, _d) = hub();
    add(&h, "Claude", BackendKind::Anthropic, &url, Some("sk-test-123"));
    let mut c = h.start(None, true).unwrap();
    let ev = run(&mut c, "How does the session look?");
    let text: String = ev.iter().filter_map(|e| if let AgentEvent::TextDelta { text } = e { Some(text.as_str()) } else { None }).collect();
    assert_eq!(text, "Checking Nothing is open.");
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::ToolFinished { tool, ok: true, .. } if tool == "core_session_show")));
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::ToolFinished { tool, ok: true, .. } if tool == "core_targets_list")));
    let AgentEvent::TurnDone { backend, input_tokens, output_tokens, cost_usd, .. } = ev.last().unwrap().clone() else { panic!() };
    assert_eq!((backend.as_str(), input_tokens, output_tokens), ("Claude", 300, 50));
    assert!(cost_usd.unwrap() > 0.0);

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    let r1 = &seen[0];
    assert_eq!(r1.path, "/v1/messages");
    assert_eq!(r1.header("x-api-key"), Some("sk-test-123"));
    assert_eq!(r1.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(r1.header("anthropic-beta"), Some("server-side-fallback-2026-07-01"));
    assert_eq!(r1.header("user-agent"), Some("CraftAnything-dev"));
    let b = &r1.body;
    assert_eq!(b["model"], "claude-opus-5-5");
    assert_eq!(b["stream"], true);
    assert_eq!(b["max_tokens"], 64000);
    assert_eq!(b["thinking"]["type"], "adaptive");
    assert_eq!(b["output_config"]["effort"], "high");
    assert_eq!(b["fallbacks"], "default");
    assert_eq!(b["tool_choice"]["type"], "auto");
    assert_eq!(b["system"][0]["cache_control"]["type"], "ephemeral");
    assert!(b["system"][0]["text"].as_str().unwrap().contains("Investigation so far"));
    let tools = b["tools"].as_array().unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted, "tool order is deterministic");
    assert!(names.contains(&"core_session_show") && names.contains(&"enable_toolset") && names.contains(&"core_handoff_write"));
    assert!(!names.contains(&"native_cursor_show"), "non-Core tools wait for enable_toolset");
    assert_eq!(tools.last().unwrap()["cache_control"]["type"], "ephemeral");
    assert!(tools.iter().all(|t| t["input_schema"]["type"] == "object"));
    // Turn 2 replays thinking (with signature) and returns BOTH results in ONE user message.
    let msgs = seen[1].body["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["content"][0], json!({"type":"thinking","thinking":"plan","signature":"SIG"}));
    let results = msgs[2]["content"].as_array().unwrap();
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["tool_use_id"], "tu_1");
    assert_eq!(results[1]["tool_use_id"], "tu_2");
    assert_eq!(results[0]["is_error"], false);
    assert!(results[0]["content"].as_str().unwrap().contains("deep"), "test sessions are in Deep mode");
    // the key is never stored in settings
    let saved = std::fs::read_to_string(_d.path().join("config.toml")).unwrap();
    assert!(!saved.contains("sk-test-123"));
}

#[test]
fn anthropic_refusal_and_http_errors() {
    let refusal = sse(&[
        json!({"type":"message_start","message":{"model":"claude-opus-5-5","usage":{"input_tokens":5}}}),
        json!({"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}),
        json!({"type":"message_stop"}),
    ]);
    let (url, _seen) = mock(vec![
        (200, "text/event-stream", refusal),
        (401, "application/json", r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#.into()),
    ]);
    let (h, _d) = hub();
    add(&h, "Claude", BackendKind::Anthropic, &url, Some("sk-test-auth"));
    let mut c = h.start(Some("Claude"), false).unwrap();
    let ev = run(&mut c, "q");
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::Error { message, .. } if message.contains("declined"))));
    let ev = run(&mut c, "q2");
    assert!(
        ev.iter().any(|e| matches!(e, AgentEvent::Error { message, remediation: Some(r), .. } if message.contains("invalid x-api-key") && r.contains("key"))),
        "{ev:?}"
    );
}

#[test]
fn openai_streaming_tool_loop() {
    let t1 = sse(&[json!({"type":"response.completed", "response":{
        "status":"completed", "model":"gpt-x", "output":[{"type":"function_call", "call_id":"call_a", "name":"core_session_show", "arguments":"{}"}],
        "usage":{"input_tokens":20,"output_tokens":3}
    }})]);
    let t2 = sse(&[
        json!({"type":"response.output_text.delta", "delta":"Nothing open."}),
        json!({"type":"response.completed", "response":{"status":"completed", "output":[{"type":"message", "content":[{"type":"output_text", "text":"Nothing open."}]}]}}),
    ]);
    let (url, seen) = mock(vec![(200, "text/event-stream", t1), (200, "text/event-stream", t2)]);
    let (h, _d) = hub();
    add(&h, "GPT", BackendKind::Openai, &url, Some("sk-o"));
    let mut c = h.start(Some("GPT"), true).unwrap();
    let ev = run(&mut c, "status?");
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::TextDelta { text } if text == "Nothing open.")));
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::ToolFinished { ok: true, .. })));
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].path, "/responses");
    assert_eq!(seen[0].header("authorization"), Some("Bearer sk-o"));
    assert_eq!(seen[0].body["store"], false);
    let input = seen[1].body["input"].as_array().unwrap();
    assert!(input.iter().any(|v| v["type"] == "function_call_output" && v["call_id"] == "call_a"));
}

#[test]
fn gemini_and_xai_use_native_tools_without_a_cli() {
    for kind in [BackendKind::Gemini, BackendKind::Xai] {
        direct_provider_tool_loop(kind);
    }
}

fn direct_provider_tool_loop(kind: BackendKind) {
    let t1 = chat_sse(&[
        json!({"model":"gpt-x","choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"core_session_show","arguments":""}}]}}]}),
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}),
        json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        json!({"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":3}}),
    ]);
    let t2 = chat_sse(&[json!({"choices":[{"delta":{"content":"Nothing open."},"finish_reason":"stop"}]})]);
    let (url, seen) = mock(vec![(200, "text/event-stream", t1), (200, "text/event-stream", t2)]);
    let (h, _d) = hub();
    add(&h, "GPT", kind, &url, Some("sk-o"));
    let mut c = h.start(Some("GPT"), true).unwrap();
    let ev = run(&mut c, "status?");
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::TextDelta { text } if text == "Nothing open.")));
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].path, "/chat/completions");
    assert_eq!(seen[0].header("authorization"), Some("Bearer sk-o"));
    assert_eq!(seen[0].body["stream_options"]["include_usage"], true);
    assert_eq!(seen[0].body["tool_choice"], "auto");
    let msgs = seen[1].body["messages"].as_array().unwrap();
    assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_a");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "call_a");
}

#[test]
fn discovery_persists_metadata_and_routes_without_exposing_keys() {
    let (url, _) = mock(vec![(
        200,
        "application/json",
        json!({"data":[
            {"id":"cheap", "context_length":32000, "supported_parameters":["tools"], "pricing":{"prompt":"0.000001", "completion":"0.000002"}},
            {"id":"expensive", "pricing":{"prompt":"0.001", "completion":"0.002"}}
        ]})
        .to_string(),
    )]);
    let (hub, dir) = hub();
    add(&hub, "Router", BackendKind::Openrouter, &url, Some("sk-test"));
    let models = hub.discover_models("Router").unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].input_per_mtok, Some(1.0));
    hub.set_routing_strategy(registry::Strategy::LowestCost).unwrap();
    let selected = hub.route(Role::Lead, &registry::Requirements { input_tokens: 1000, output_tokens: 1000, tools: true, ..Default::default() }).unwrap();
    assert_eq!(selected.config.model, "cheap");
    assert_eq!(hub.config().model_catalog, models);
    let saved = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
    assert!(!saved.contains("sk-test"));
    hub.set_enabled("Router", false).unwrap();
    assert!(hub.recommend_model(registry::Strategy::Auto, &Default::default()).is_err());
}

#[test]
fn local_server_without_tools_falls_back_to_react() {
    let t1 = chat_sse(&[json!({"choices":[{"delta":{"content":"Thought: check\nAction: core_session_show\nAction Input: {}"},"finish_reason":"stop"}]})]);
    let t2 = chat_sse(&[json!({"choices":[{"delta":{"content":"Final Answer: nothing is open"},"finish_reason":"stop"}]})]);
    let (url, seen) = mock(vec![
        (400, "application/json", r#"{"error":{"message":"registry.ollama.ai/library/tiny does not support tools"}}"#.into()),
        (200, "text/event-stream", t1),
        (200, "text/event-stream", t2),
    ]);
    let (h, _d) = hub();
    add(&h, "Tiny", BackendKind::OpenaiCompatible, &format!("{url}/v1"), None);
    let mut c = h.start(Some("Tiny"), true).unwrap();
    let ev = run(&mut c, "status?");
    assert!(ev.iter().any(|e| matches!(e, AgentEvent::ToolFinished { tool, ok: true, .. } if tool == "core_session_show")), "{ev:?}");
    assert!(matches!(ev.last(), Some(AgentEvent::TurnDone { cost_usd: Some(c), .. }) if *c == 0.0));
    let seen = seen.lock().unwrap();
    assert!(seen[0].body.get("tools").is_some());
    assert!(seen[0].header("authorization").is_none());
    assert!(seen[1].body.get("tools").is_none());
    let last = seen[2].body["messages"].as_array().unwrap().last().unwrap().clone();
    assert!(last["content"].as_str().unwrap().starts_with("Observation:"));
}

#[test]
fn hub_config_keys_and_test_backend() {
    let t = sse(&[
        json!({"type":"message_start","message":{"model":"claude-haiku-5-5","usage":{"input_tokens":3}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"ready"}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ]);
    let models = json!({"data":[{"id":"claude-opus-5-5"},{"id":"claude-haiku-5-5"}]}).to_string();
    let (url, seen) = mock(vec![(200, "text/event-stream", t), (200, "application/json", models.clone()), (200, "application/json", models)]);
    let (h, d) = hub();
    assert!(h.backends().is_empty());
    assert!(h.start(None, true).is_err());
    add(&h, "Claude", BackendKind::Anthropic, &url, Some("k1"));
    add(&h, "Local", BackendKind::OpenaiCompatible, "http://127.0.0.1:9/v1", None);
    assert_eq!(h.backends().len(), 2);
    assert!(h.has_key("Claude") && h.has_key("Local"));
    let r = h.test_backend("Claude").unwrap();
    assert_eq!(r.reply, "ready");
    assert_eq!(r.model, "claude-haiku-5-5");
    assert_eq!(seen.lock().unwrap()[0].body["output_config"]["effort"], "high");
    assert_eq!(h.list_models("Claude").unwrap(), vec!["claude-opus-5-5", "claude-haiku-5-5"]);
    let reach = h.check_reachability();
    assert_eq!(reach.len(), 2);
    assert!(reach[0].1.is_ok(), "{reach:?}");
    assert!(reach[1].1.is_err());
    // settings persisted to the file; reload shows both
    let s = Settings::load_from(&d.path().join("config.toml"));
    assert_eq!(AiConfig::from_settings(&s).backends.len(), 2);
    assert!(h.remove_backend("Claude").unwrap());
    assert!(!h.remove_backend("Claude").unwrap());
    assert!(h.test_backend("Claude").is_err());
    // missing key is a clear error
    h.add_backend(
        BackendConfig {
            name: "NoKey".into(),
            kind: BackendKind::Openrouter,
            model: String::new(),
            base_url: Some("http://127.0.0.1:9".into()),
            ssh_host: None,
            enabled: true,
        },
        None,
    )
    .unwrap();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        let e = h.start(Some("NoKey"), true).err().unwrap();
        assert_eq!(e.code, ErrorCode::BadSettings);
    }
    assert!(h.compare("Local", "Local", true).is_err());
}
