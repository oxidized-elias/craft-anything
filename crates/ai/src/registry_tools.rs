//! Headless model discovery, recommendations and routing settings.
use crate::{
    AiConfig, KeyStore,
    registry::{self, Requirements, Strategy},
};
use craft_anything_engine::tool::{Ctx, Effects, Requires, ToolOutput, ToolSpec, Toolset, parse_params, schema_of};
use craft_anything_evidence::ToolError;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

pub fn specs() -> Vec<ToolSpec> {
    let spec = |id, title, summary, effects, input_schema, run| ToolSpec {
        id,
        title,
        summary,
        beginner: summary,
        toolset: Toolset::Setup,
        effects,
        requires: Requires::Nothing,
        input_schema,
        run,
    };
    vec![
        spec(
            "settings_ai_models_list",
            "List discovered AI models",
            "Read the saved provider model catalogs. Unknown capabilities and prices stay unknown.",
            Effects::READ_ONLY,
            schema_of::<List>,
            settings_ai_models_list,
        ),
        spec(
            "settings_ai_models_discover",
            "Discover AI models",
            "Refresh one configured API provider's catalog and save it. Never returns API keys.",
            Effects::ACCESSES_NETWORK.union(Effects::WRITES_FILESYSTEM).union(Effects::MUTATES_SESSION),
            schema_of::<Backend>,
            settings_ai_models_discover,
        ),
        spec(
            "settings_ai_model_recommend",
            "Recommend an AI model",
            "Recommend an available model using verified metadata and explicit task constraints.",
            Effects::READ_ONLY,
            schema_of::<Recommend>,
            settings_ai_model_recommend,
        ),
        spec(
            "settings_ai_routing_set",
            "Choose automatic AI routing",
            "Save the routing policy for new automatic conversations; explicit AI choices take precedence.",
            Effects::WRITES_FILESYSTEM.union(Effects::MUTATES_SESSION),
            schema_of::<Routing>,
            settings_ai_routing_set,
        ),
        spec(
            "settings_ai_model_set",
            "Choose an AI model",
            "Set the model id for one configured API backend, preserving its options and key.",
            Effects::WRITES_FILESYSTEM.union(Effects::MUTATES_SESSION),
            schema_of::<Model>,
            settings_ai_model_set,
        ),
    ]
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct List {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    limit: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Backend {
    backend: String,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Model {
    backend: String,
    model: String,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Routing {
    strategy: Strategy,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Recommend {
    #[serde(default)]
    strategy: Strategy,
    #[serde(default)]
    requirements: Requirements,
}

fn config(ctx: &Ctx<'_>) -> Result<AiConfig, ToolError> {
    AiConfig::from_table(&ctx.session.settings.ai)
}
fn save(ctx: &mut Ctx<'_>, cfg: &AiConfig) -> Result<(), ToolError> {
    let mut next = ctx.session.settings.clone();
    next.ai = cfg.to_table()?;
    next.save_to(&ctx.session.config_dir().join("config.toml"))?;
    ctx.session.settings = next;
    Ok(())
}
fn settings_ai_models_list(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: List = parse_params(params)?;
    let cfg = config(ctx)?;
    let all: Vec<_> = cfg.model_catalog.iter().filter(|m| p.backend.as_ref().is_none_or(|b| *b == m.backend)).collect();
    let models: Vec<_> = all.iter().skip(p.offset).take(p.limit.unwrap_or(200).min(2000)).collect();
    Ok(ToolOutput::Value(json!({"total":all.len(),"offset":p.offset,"models":models})))
}
fn settings_ai_models_discover(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: Backend = parse_params(params)?;
    let mut cfg = config(ctx)?;
    let entry = cfg.get(&p.backend).ok_or_else(|| ToolError::invalid("Unknown AI backend."))?;
    let keys = ctx.session.ext.get::<KeyStore>().map(|k| k.as_ref().clone()).unwrap_or_else(KeyStore::system);
    let models = registry::discover(&entry.config, keys.resolve(&p.backend, entry.config.kind))?;
    cfg.model_catalog.retain(|m| m.backend != p.backend);
    cfg.model_catalog.extend(models.clone());
    save(ctx, &cfg)?;
    Ok(ToolOutput::Value(json!({"backend":p.backend,"total":models.len(),"models":models.iter().take(200).collect::<Vec<_>>()})))
}
fn settings_ai_model_recommend(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: Recommend = parse_params(params)?;
    let cfg = config(ctx)?;
    let models: Vec<_> = cfg.model_catalog.iter().filter(|m| cfg.get(&m.backend).is_some_and(|b| b.config.enabled)).cloned().collect();
    Ok(ToolOutput::Value(json!({"model":registry::recommend(&models, p.strategy, &p.requirements)?})))
}
fn settings_ai_routing_set(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: Routing = parse_params(params)?;
    let mut cfg = config(ctx)?;
    cfg.routing_strategy = p.strategy;
    save(ctx, &cfg)?;
    Ok(ToolOutput::Value(json!({"strategy":p.strategy})))
}
fn settings_ai_model_set(ctx: &mut Ctx<'_>, params: Value) -> Result<ToolOutput, ToolError> {
    let p: Model = parse_params(params)?;
    if p.model.trim().is_empty() {
        return Err(ToolError::invalid("The model id cannot be empty."));
    }
    let mut cfg = config(ctx)?;
    let mut entry = cfg.get(&p.backend).cloned().ok_or_else(|| ToolError::invalid("Unknown AI backend."))?;
    if entry.config.kind == crate::BackendKind::AgentCli {
        return Err(ToolError::invalid("Choose an API backend."));
    }
    entry.config.model = p.model.clone();
    cfg.upsert(entry);
    save(ctx, &cfg)?;
    Ok(ToolOutput::Value(json!({"backend":p.backend,"model":p.model})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use craft_anything_engine::{CallOptions, Caller, Engine};
    #[test]
    fn routing_tools_work_headlessly_in_clean_room() {
        let mut session = craft_anything_engine::testing::session();
        session.settings.deep_mode = false;
        let engine = Engine::shared(session, specs());
        let call = || CallOptions::new(Caller::Agent);
        engine.call("settings_ai_routing_set", json!({"strategy":"LOCAL_FIRST"}), call()).unwrap();
        let cfg = AiConfig::from_table(&engine.session().settings.ai).unwrap();
        assert_eq!(cfg.routing_strategy, Strategy::LocalFirst);
        assert!(engine.call("settings_ai_models_list", json!({}), call()).is_ok());
        assert!(engine.call("settings_ai_model_recommend", json!({"requirements":{"tools":true}}), call()).is_err());
    }
}
