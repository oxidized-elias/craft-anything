//! Unified discovery metadata and deterministic, constraint-aware routing.
//! Missing provider metadata stays unknown: model names never imply price or free quota.
use crate::{BackendConfig, BackendKind};
use craft_anything_evidence::ToolError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One live catalog request, shared by the hub and engine tools.
pub fn discover(c: &BackendConfig, key: Option<String>) -> Result<Vec<ModelInfo>, ToolError> {
    if c.kind == BackendKind::AgentCli {
        return Err(ToolError::invalid("Model discovery requires an API backend."));
    }
    if c.ssh_host.as_deref().is_some_and(|h| !h.trim().is_empty()) {
        return Err(ToolError::unavailable("Remote model discovery is not supported yet. Type the model id instead."));
    }
    let base = c.base_url.as_deref().filter(|b| !b.trim().is_empty());
    let timeout = std::time::Duration::from_secs(10);
    let required = || key.clone().ok_or_else(|| ToolError::invalid("This provider needs an API key. Add it in Settings → AI."));
    let providers = crate::providers::Providers::default();
    let raw = if let Some(provider) = providers.get(c.kind) {
        provider.catalog(base, required()?, timeout)?
    } else if c.kind == BackendKind::Openrouter {
        crate::openai::OpenAi::openrouter(base, required()?, &c.model).model_catalog(timeout)?
    } else if c.kind == BackendKind::OpenaiCompatible {
        crate::openai::OpenAi::compatible(&c.name, base.unwrap_or("http://127.0.0.1:11434/v1"), key, &c.model).model_catalog(timeout)?
    } else if c.kind == BackendKind::Openai {
        crate::openai::OpenAi::official(base, required()?, &c.model).model_catalog(timeout)?
    } else {
        crate::anthropic::Anthropic::new(base, required()?, &c.model).list_models(timeout)?.into_iter().map(|id| serde_json::json!({"id":id})).collect()
    };
    Ok(raw.iter().take(10_000).filter_map(|v| ModelInfo::from_json(c, v)).collect())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Strategy {
    #[default]
    Auto,
    FreeTierFirst,
    LowestCost,
    Fastest,
    BestQuality,
    LocalFirst,
}

impl Strategy {
    pub const ALL: [Self; 6] = [Self::Auto, Self::FreeTierFirst, Self::LowestCost, Self::Fastest, Self::BestQuality, Self::LocalFirst];
    pub fn title(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::FreeTierFirst => "Free tier first",
            Self::LowestCost => "Lowest cost",
            Self::Fastest => "Fastest",
            Self::BestQuality => "Best quality",
            Self::LocalFirst => "Local first",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelInfo {
    pub backend: String,
    pub id: String,
    pub context_tokens: Option<u64>,
    pub tools: Option<bool>,
    pub vision: Option<bool>,
    /// Account eligibility and remaining quotas are not implied by zero list price.
    pub free_tier: Option<bool>,
    pub input_per_mtok: Option<f64>,
    pub output_per_mtok: Option<f64>,
    pub latency_ms: Option<u64>,
    /// An explicit user or evaluator score, never guessed from the model id.
    pub quality: Option<u32>,
    pub available: bool,
    pub local: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Requirements {
    pub tools: bool,
    pub vision: bool,
    pub context_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub max_cost_usd: Option<f64>,
}

impl Default for Requirements {
    fn default() -> Self {
        Self { tools: false, vision: false, context_tokens: 0, input_tokens: 1000, output_tokens: 1000, max_cost_usd: None }
    }
}

fn rate(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    let n = value.as_f64().or_else(|| value.as_str()?.parse().ok())?;
    (n.is_finite() && n >= 0.0).then_some(n)
}

impl ModelInfo {
    pub fn from_json(backend: &BackendConfig, value: &Value) -> Option<Self> {
        let id = value.get("id").or_else(|| value.get("name"))?.as_str()?.trim_start_matches("models/").to_string();
        if id.is_empty() {
            return None;
        }
        let supported = value.get("supported_parameters").and_then(Value::as_array);
        let modalities = value.pointer("/architecture/input_modalities").and_then(Value::as_array);
        let local = is_local(backend);
        Some(Self {
            backend: backend.name.clone(),
            id,
            local,
            available: !value.get("deprecated").and_then(Value::as_bool).unwrap_or(false),
            context_tokens: value.get("context_length").or_else(|| value.get("inputTokenLimit")).and_then(Value::as_u64),
            tools: supported.map(|a| a.iter().any(|v| v.as_str() == Some("tools"))),
            vision: modalities.map(|a| a.iter().any(|v| v.as_str() == Some("image"))),
            input_per_mtok: if local { Some(0.0) } else { rate(value.pointer("/pricing/prompt")).map(|n| n * 1_000_000.0).filter(|n| n.is_finite()) },
            output_per_mtok: if local { Some(0.0) } else { rate(value.pointer("/pricing/completion")).map(|n| n * 1_000_000.0).filter(|n| n.is_finite()) },
            ..Self::default()
        })
    }

    pub fn estimated_cost(&self, req: &Requirements) -> Option<f64> {
        let cost = self.input_per_mtok? * req.input_tokens as f64 / 1_000_000.0 + self.output_per_mtok? * req.output_tokens as f64 / 1_000_000.0;
        (cost.is_finite() && cost >= 0.0).then_some(cost)
    }

    fn accepts(&self, req: &Requirements) -> bool {
        self.available
            && (!req.tools || self.tools == Some(true))
            && (!req.vision || self.vision == Some(true))
            && (req.context_tokens == 0 || self.context_tokens.is_some_and(|c| c >= req.context_tokens))
            && req.max_cost_usd.is_none_or(|limit| limit.is_finite() && limit >= 0.0 && self.estimated_cost(req).is_some_and(|c| c <= limit))
    }
}

/// Only loopback servers are treated as free local inference. A hosted compatible endpoint
/// can charge money just like any other cloud provider.
pub fn is_local(backend: &BackendConfig) -> bool {
    if backend.kind != BackendKind::OpenaiCompatible || backend.ssh_host.as_deref().is_some_and(|h| !h.is_empty()) {
        return false;
    }
    let base = backend.base_url.as_deref().filter(|b| !b.trim().is_empty()).unwrap_or("http://127.0.0.1:11434/v1");
    let Some((scheme, rest)) = base.split_once("://") else {
        return false;
    };
    if scheme != "http" && scheme != "https" {
        return false;
    }
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.contains('@') {
        return false;
    }
    ["localhost", "127.0.0.1", "[::1]"]
        .iter()
        .any(|host| authority == *host || authority.strip_prefix(host).is_some_and(|s| s.strip_prefix(':').is_some_and(|p| p.parse::<u16>().is_ok())))
}

pub fn recommend<'a>(models: &'a [ModelInfo], strategy: Strategy, req: &Requirements) -> Result<&'a ModelInfo, ToolError> {
    let score = |model: &ModelInfo| -> Option<f64> {
        match strategy {
            Strategy::Auto => Some(0.0),
            Strategy::FreeTierFirst => model.free_tier.map(|free| if free { 0.0 } else { 1.0 }),
            Strategy::LowestCost => model.estimated_cost(req),
            Strategy::Fastest => model.latency_ms.map(|n| n as f64),
            Strategy::BestQuality => model.quality.map(|n| -(n as f64)),
            Strategy::LocalFirst => Some(if model.local { 0.0 } else { 1.0 }),
        }
    };
    models
        .iter()
        .filter(|m| m.accepts(req))
        .min_by(|a, b| match (score(a), score(b)) {
            (Some(a), Some(b)) => a.total_cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        })
        .ok_or_else(|| {
            ToolError::unavailable("No discovered model satisfies the task constraints.")
                .with_remediation("Refresh the model catalog or supply verified capability and pricing metadata.")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(id: &str) -> ModelInfo {
        ModelInfo { id: id.into(), available: true, ..Default::default() }
    }
    #[test]
    fn unknown_capabilities_and_prices_do_not_satisfy_constraints() {
        let models = vec![model("unknown")];
        assert!(recommend(&models, Strategy::Auto, &Requirements { tools: true, ..Default::default() }).is_err());
        assert!(recommend(&models, Strategy::LowestCost, &Requirements { max_cost_usd: Some(1.0), ..Default::default() }).is_err());
    }
    #[test]
    fn all_strategies_respect_requirements() {
        let models = vec![
            model("unknown"),
            ModelInfo {
                local: true,
                tools: Some(true),
                free_tier: Some(true),
                input_per_mtok: Some(0.0),
                output_per_mtok: Some(0.0),
                latency_ms: Some(10),
                quality: Some(9),
                ..model("capable")
            },
        ];
        for strategy in Strategy::ALL {
            assert_eq!(recommend(&models, strategy, &Requirements { tools: true, ..Default::default() }).unwrap().id, "capable");
        }
    }
    #[test]
    fn provider_metadata_remains_honest() {
        let b = BackendConfig { name: "cloud".into(), kind: BackendKind::Openrouter, model: "m".into(), base_url: None, ssh_host: None, enabled: true };
        let m = ModelInfo::from_json(
            &b,
            &serde_json::json!({"id":"m", "pricing":{"prompt":"0", "completion":"0"},
            "supported_parameters":["tools"], "architecture":{"input_modalities":["text","image"]}, "context_length":32000}),
        )
        .unwrap();
        assert_eq!(m.free_tier, None);
        assert_eq!(m.tools, Some(true));
        assert_eq!(m.vision, Some(true));
        assert_eq!(m.input_per_mtok, Some(0.0));
        let mut b = b;
        b.kind = BackendKind::OpenaiCompatible;
        for base in ["https://hosted.example/v1", "http://localhost.evil/v1", "http://localhost@evil/v1"] {
            b.base_url = Some(base.into());
            assert!(!is_local(&b));
        }
        b.base_url = Some("http://[::1]:1234/v1".into());
        assert!(is_local(&b));
    }
}
