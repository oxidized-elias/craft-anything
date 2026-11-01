//! OpenAI Responses transport. The native loop owns tools and conversation persistence.
use crate::{http, model::*};
use craft_anything_evidence::{ErrorCode, ToolError};
use serde_json::{Value, json};

pub struct Responses {
    client: crate::openai::OpenAi,
}

impl Responses {
    pub fn new(base: Option<&str>, key: String, model: &str) -> Self {
        Self { client: crate::openai::OpenAi::official(base, key, model) }
    }

    pub fn body(&self, req: &TurnRequest<'_>) -> Value {
        let mut input = Vec::new();
        for message in req.messages {
            for block in &message.blocks {
                match block {
                    Block::Text { text } => input.push(json!({
                        "role": if message.role == Speaker::Assistant { "assistant" } else { "user" },
                        "content": text,
                    })),
                    Block::ToolUse { id, name, input: args } => input.push(json!({
                        "type": "function_call", "call_id": id, "name": name, "arguments": args.to_string(),
                    })),
                    Block::ToolResult { tool_use_id, content, .. } => input.push(json!({
                        "type": "function_call_output", "call_id": tool_use_id, "output": content,
                    })),
                    Block::ProviderData { provider, data } => {
                        if provider == "openai_responses" && data.get("type").and_then(Value::as_str) == Some("reasoning") {
                            input.push(data.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function", "name": t.name, "description": t.description,
                    "parameters": t.schema, "strict": false,
                })
            })
            .collect();
        json!({
            "model": self.client.model, "instructions": req.system, "input": input,
            "tools": tools, "stream": true, "store": false,
            "include": ["reasoning.encrypted_content"], "max_output_tokens": req.max_tokens,
        })
    }
}

impl ChatModel for Responses {
    fn model(&self) -> String {
        self.client.model.clone()
    }

    fn turn(&mut self, req: &TurnRequest<'_>, on_delta: &mut dyn FnMut(Delta), stop: &Stop) -> Result<TurnResult, ToolError> {
        stop.check()?;
        let url = format!("{}/responses", self.client.base_url);
        let reader = http::post_json(&http::stream_agent(), "OpenAI", &url, &self.client.headers(), &self.body(req))?;
        let mut result = None;
        http::parse_sse(reader, |event| {
            stop.check()?;
            if event.data.trim().is_empty() {
                return Ok(true);
            }
            let v: Value = serde_json::from_str(&event.data).map_err(|_| ToolError::new(ErrorCode::GarbledOutput, "OpenAI sent an invalid stream event."))?;
            match v.get("type").and_then(Value::as_str).unwrap_or_default() {
                "response.output_text.delta" => {
                    if let Some(text) = v.get("delta").and_then(Value::as_str) {
                        on_delta(Delta::Text(text.into()));
                    }
                }
                "response.completed" | "response.incomplete" => {
                    result = Some(parse_response(v.get("response").unwrap_or(&Value::Null), &self.client.model)?);
                    return Ok(false);
                }
                "error" | "response.failed" => {
                    return Err(ToolError::new(ErrorCode::EngineMissing, "OpenAI could not complete the response.")
                        .with_remediation("Check provider availability, the model and account limits, or choose another backend."));
                }
                _ => {}
            }
            Ok(true)
        })?;
        stop.check()?;
        result.ok_or_else(|| ToolError::new(ErrorCode::Truncated, "OpenAI's response stream ended before completion."))
    }
}

fn parse_response(response: &Value, fallback: &str) -> Result<TurnResult, ToolError> {
    if !matches!(response.get("status").and_then(Value::as_str), Some("completed" | "incomplete")) || response.get("output").and_then(Value::as_array).is_none()
    {
        return Err(ToolError::new(ErrorCode::GarbledOutput, "OpenAI returned an invalid terminal response."));
    }
    let mut blocks = Vec::new();
    let mut refused = false;
    let mut call_ids = std::collections::BTreeSet::new();
    for item in response.get("output").and_then(Value::as_array).into_iter().flatten() {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "reasoning" => blocks.push(Block::ProviderData { provider: "openai_responses".into(), data: item.clone() }),
            "function_call" => {
                let id = item.get("call_id").and_then(Value::as_str).filter(|v| !v.is_empty());
                let name = item.get("name").and_then(Value::as_str).filter(|v| !v.is_empty());
                let args = item.get("arguments").and_then(Value::as_str).unwrap_or_default();
                let input: Value = serde_json::from_str(args)
                    .map_err(|_| ToolError::new(ErrorCode::GarbledOutput, "OpenAI returned invalid function arguments; no tools were run."))?;
                let (Some(id), Some(name)) = (id, name) else {
                    return Err(ToolError::new(ErrorCode::GarbledOutput, "OpenAI returned a function call without its name or id."));
                };
                if call_ids.len() >= 256 || !call_ids.insert(id.to_string()) {
                    return Err(ToolError::new(ErrorCode::GarbledOutput, "OpenAI returned too many calls or duplicate call ids."));
                }
                if !input.is_object() {
                    return Err(ToolError::new(ErrorCode::GarbledOutput, "Function arguments must be an object."));
                }
                blocks.push(Block::ToolUse { id: id.into(), name: name.into(), input });
            }
            "message" => {
                for part in item.get("content").and_then(Value::as_array).into_iter().flatten() {
                    if part.get("type").and_then(Value::as_str) == Some("refusal") {
                        refused = true;
                    }
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        blocks.push(Block::Text { text: text.into() });
                    }
                }
            }
            _ => {}
        }
    }
    let stop = if refused || response.pointer("/incomplete_details/reason").and_then(Value::as_str) == Some("content_filter") {
        StopReason::Refusal
    } else if response.get("status").and_then(Value::as_str) == Some("incomplete") {
        StopReason::MaxTokens
    } else if blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })) {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    };
    let usage = response.get("usage").unwrap_or(&Value::Null);
    let cached = usage.pointer("/input_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0);
    Ok(TurnResult {
        message: Msg { role: Speaker::Assistant, blocks },
        stop,
        model: response.get("model").and_then(Value::as_str).unwrap_or(fallback).into(),
        usage: Usage {
            input_tokens: usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0).saturating_sub(cached),
            output_tokens: usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
            cache_read_tokens: cached,
            cache_write_tokens: 0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calls_usage_and_reasoning_survive_replay() {
        let response = json!({"status":"completed", "model":"test", "output":[
            {"type":"reasoning", "id":"r1", "summary":[], "encrypted_content":"opaque"},
            {"type":"function_call", "call_id":"c1", "name":"core_session_show", "arguments":"{}"}
        ], "usage":{"input_tokens":100, "output_tokens":12, "input_tokens_details":{"cached_tokens":40}}});
        let r = parse_response(&response, "fallback").unwrap();
        assert_eq!(r.stop, StopReason::ToolUse);
        assert_eq!(r.usage.input_tokens, 60);
        assert_eq!(r.usage.cache_read_tokens, 40);
        let messages =
            vec![r.message, Msg { role: Speaker::User, blocks: vec![Block::ToolResult { tool_use_id: "c1".into(), content: "done".into(), is_error: false }] }];
        let b = Responses::new(None, "sk-test".into(), "test").body(&TurnRequest {
            system: "s",
            messages: &messages,
            tools: &[],
            effort: Effort::Medium,
            max_tokens: 1024,
        });
        assert_eq!(b["input"][0]["encrypted_content"], "opaque");
        assert_eq!(b["input"][1]["call_id"], "c1");
        assert_eq!(b["input"][2]["type"], "function_call_output");
        assert_eq!(b["store"], false);
    }
    #[test]
    fn malformed_calls_are_not_executable() {
        assert!(parse_response(&json!({"status":"completed","output":[{"type":"function_call","arguments":"{"}]}), "m").is_err());
        let r = parse_response(&json!({"status":"incomplete","output":[]}), "m").unwrap();
        assert_eq!(r.stop, StopReason::MaxTokens);
    }
}
