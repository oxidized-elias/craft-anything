//! Direct API providers registered independently of the agent harness.
use crate::{BackendKind, model::ChatModel, openai::OpenAi};
use craft_anything_evidence::ToolError;
use std::sync::Arc;
use std::time::Duration;

/// A provider supplies transport and discovery; execution stays in the native agent loop.
pub trait Provider: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn title(&self) -> &'static str;
    fn key_env(&self) -> &'static str;
    fn models(&self) -> &'static [&'static str];
    fn connect(&self, base: Option<&str>, key: String, model: &str) -> Box<dyn ChatModel>;
    fn discover(&self, base: Option<&str>, key: String, timeout: Duration) -> Result<Vec<String>, ToolError>;
    fn catalog(&self, base: Option<&str>, key: String, timeout: Duration) -> Result<Vec<serde_json::Value>, ToolError> {
        Ok(self.discover(base, key, timeout)?.into_iter().map(|id| serde_json::json!({"id":id})).collect())
    }
}

/// Forks can register additional implementations without changing the execution loop.
#[derive(Clone)]
pub struct Providers(pub Vec<Arc<dyn Provider>>);

impl Default for Providers {
    fn default() -> Self {
        Self(vec![
            Arc::new(CompatibleProvider {
                kind: BackendKind::Gemini,
                title: "Google Gemini API",
                key_env: "GEMINI_API_KEY",
                base: "https://generativelanguage.googleapis.com/v1beta/openai",
                models: &["gemini-3.8-flash"],
            }),
            Arc::new(CompatibleProvider {
                kind: BackendKind::Xai,
                title: "xAI Grok API",
                key_env: "XAI_API_KEY",
                base: "https://api.x.ai/v1",
                models: &["grok-4.7"],
            }),
        ])
    }
}

impl Providers {
    pub fn get(&self, kind: BackendKind) -> Option<&dyn Provider> {
        self.0.iter().find(|p| p.kind() == kind).map(AsRef::as_ref)
    }
}

struct CompatibleProvider {
    kind: BackendKind,
    title: &'static str,
    key_env: &'static str,
    base: &'static str,
    models: &'static [&'static str],
}

impl CompatibleProvider {
    fn client(&self, base: Option<&str>, key: String, model: &str) -> OpenAi {
        let mut client = OpenAi::compatible(self.title, base.unwrap_or(self.base), Some(key), model);
        // Hosted APIs must support real function calls; never silently downgrade permissions
        // and tool structure to a text protocol.
        client.auto_react = false;
        client
    }
}

impl Provider for CompatibleProvider {
    fn kind(&self) -> BackendKind {
        self.kind
    }
    fn title(&self) -> &'static str {
        self.title
    }
    fn key_env(&self) -> &'static str {
        self.key_env
    }
    fn models(&self) -> &'static [&'static str] {
        self.models
    }
    fn connect(&self, base: Option<&str>, key: String, model: &str) -> Box<dyn ChatModel> {
        Box::new(self.client(base, key, model))
    }
    fn discover(&self, base: Option<&str>, key: String, timeout: Duration) -> Result<Vec<String>, ToolError> {
        self.client(base, key, "").list_models(timeout)
    }
    fn catalog(&self, base: Option<&str>, key: String, timeout: Duration) -> Result<Vec<serde_json::Value>, ToolError> {
        if self.kind == BackendKind::Gemini && base.is_none() {
            let deadline = std::time::Instant::now() + timeout;
            let headers = [("x-goog-api-key", key)];
            let mut models = Vec::new();
            let mut page = String::new();
            for _ in 0..20 {
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Err(ToolError::new(craft_anything_evidence::ErrorCode::TimedOut, "Gemini model discovery timed out."));
                }
                let agent = crate::http::quick_agent(remaining);
                let url = format!("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000{page}");
                let response = crate::http::get_json(&agent, self.title, &url, &headers)?;
                let entries = response
                    .get("models")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| ToolError::invalid("Gemini returned no model catalog array."))?;
                models.extend(
                    entries
                        .iter()
                        .filter(|m| {
                            m.get("supportedGenerationMethods")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|methods| methods.iter().any(|v| v.as_str() == Some("generateContent")))
                        })
                        .cloned(),
                );
                if models.len() > 10_000 {
                    return Err(ToolError::invalid("Gemini's model catalog exceeded the model limit."));
                }
                let Some(token) = response.get("nextPageToken").and_then(serde_json::Value::as_str).filter(|t| !t.is_empty()) else {
                    return Ok(models);
                };
                if token.len() > 8192 {
                    return Err(ToolError::invalid("Gemini returned an oversized page token."));
                }
                let encoded: String = token
                    .bytes()
                    .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { char::from(b).to_string() } else { format!("%{b:02X}") })
                    .collect();
                page = format!("&pageToken={encoded}");
            }
            return Err(ToolError::invalid("Gemini's model catalog exceeded the page limit."));
        }
        self.client(base, key, "").model_catalog(timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_providers_are_registered() {
        let all = Providers::default();
        for kind in [BackendKind::Gemini, BackendKind::Xai] {
            let p = all.get(kind).unwrap();
            assert!(!p.models().is_empty());
            assert!(p.key_env().ends_with("_API_KEY"));
            assert_eq!(p.connect(None, "sk-test".into(), "test-model").model(), "test-model");
        }
    }
}
