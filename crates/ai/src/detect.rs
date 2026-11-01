//! Find AI that's ready to use on this machine: env keys, local servers, agent CLIs, ssh hosts.
//! Probes run in parallel and the whole scan is bounded (about a second).

use crate::openai::{LOCAL_SERVERS, OpenAi};
use crate::{BackendConfig, BackendKind, Detected};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Per-server connect bound.
pub const PROBE: Duration = Duration::from_millis(300);
/// Whole-scan bound.
pub const TOTAL: Duration = Duration::from_millis(1500);

fn cfg(name: &str, kind: BackendKind, model: &str) -> BackendConfig {
    BackendConfig { name: name.into(), kind, model: model.into(), base_url: None, ssh_host: None, enabled: true }
}

/// Environment keys that are set.
pub fn env_keys(get: impl Fn(&str) -> Option<String>) -> Vec<Detected> {
    let mut out = Vec::new();
    for (var, name, kind, model) in [
        ("ANTHROPIC_API_KEY", "Claude", BackendKind::Anthropic, crate::anthropic::DEFAULT_MODEL),
        ("OPENAI_API_KEY", "OpenAI", BackendKind::Openai, crate::openai::OPENAI_DEFAULT_MODEL),
        ("OPENROUTER_API_KEY", "OpenRouter", BackendKind::Openrouter, crate::openai::OPENROUTER_DEFAULT_MODEL),
    ] {
        if get(var).is_some_and(|v| !v.trim().is_empty()) {
            out.push(Detected { suggestion: cfg(name, kind, model), reason: format!("Found {var} in your environment"), needs_key: false });
        }
    }
    for p in crate::providers::Providers::default().0 {
        let found = get(p.key_env()).is_some_and(|v| !v.trim().is_empty());
        out.push(Detected {
            suggestion: cfg(p.title(), p.kind(), p.models().first().copied().unwrap_or_default()),
            reason: if found {
                format!("Found {} in your environment", p.key_env())
            } else {
                "Build and answer with an API key; no coding agent installation needed.".into()
            },
            needs_key: !found,
        });
    }
    // Kimi K3 builds through OpenRouter: with its key found, or one the person pastes.
    let found = get("OPENROUTER_API_KEY").is_some_and(|v| !v.trim().is_empty());
    out.push(Detected {
        suggestion: cfg(KIMI_NAME, BackendKind::Openrouter, crate::openai::KIMI_K3),
        reason: if found {
            "Kimi K3 through OpenRouter, with the OPENROUTER_API_KEY in your environment".into()
        } else {
            "Kimi K3 through OpenRouter: paste your OpenRouter key".into()
        },
        needs_key: !found,
    });
    out
}

/// The name Kimi K3 is added under.
pub const KIMI_NAME: &str = "Kimi K3 (OpenRouter)";

/// Is something listening at this URL's host:port? (bounded TCP connect)
pub fn port_open(base_url: &str, timeout: Duration) -> bool {
    let hostport = base_url.split("://").nth(1).unwrap_or(base_url).split('/').next().unwrap_or_default();
    let addrs: Vec<SocketAddr> = hostport.to_socket_addrs().map(|a| a.collect()).unwrap_or_default();
    addrs.iter().any(|a| TcpStream::connect_timeout(a, timeout).is_ok())
}

/// One local server: connect, then list models.
pub fn probe_local(label: &str, base: &str) -> Option<Detected> {
    if !port_open(base, PROBE) {
        return None;
    }
    let models = OpenAi::compatible(label, base, None, "").list_models(Duration::from_millis(700)).unwrap_or_default();
    let model = models.first().cloned().unwrap_or_default();
    let reason = match models.len() {
        0 => format!("Found {label} running at {base}"),
        1 => format!("Found {label} running with 1 model"),
        n => format!("Found {label} running with {n} models"),
    };
    let mut c = cfg(&format!("{label} (local)"), BackendKind::OpenaiCompatible, &model);
    c.base_url = Some(base.to_string());
    Some(Detected { suggestion: c, reason, needs_key: false })
}

/// Agent CLIs on PATH.
pub fn agent_clis() -> Vec<Detected> {
    crate::cli_agents::KNOWN
        .iter()
        .filter_map(|(id, prog, display)| {
            craft_anything_platform::which(prog).map(|p| Detected {
                suggestion: cfg(display, BackendKind::AgentCli, id),
                reason: format!("Found {display} at {}", p.display()),
                needs_key: false,
            })
        })
        .collect()
}

/// Everything, bounded by [`TOTAL`].
pub fn detect_all() -> Vec<Detected> {
    let start = Instant::now();
    let (tx, rx) = mpsc::channel::<(usize, Vec<Detected>)>();
    let mut jobs = 0usize;
    for (i, (label, base)) in LOCAL_SERVERS.iter().enumerate() {
        let tx = tx.clone();
        if std::thread::Builder::new()
            .name("ai-detect".into())
            .spawn(move || {
                let _ = tx.send((i + 10, probe_local(label, base).into_iter().collect()));
            })
            .is_ok()
        {
            jobs += 1;
        }
    }
    {
        let tx = tx.clone();
        if std::thread::Builder::new()
            .name("ai-detect".into())
            .spawn(move || {
                let _ = tx.send((5, agent_clis()));
            })
            .is_ok()
        {
            jobs += 1;
        }
    }
    drop(tx);
    let mut parts: Vec<(usize, Vec<Detected>)> = vec![(0, env_keys(|v| std::env::var(v).ok()))];
    while jobs > 0 {
        let left = TOTAL.saturating_sub(start.elapsed());
        match rx.recv_timeout(left) {
            Ok(p) => {
                parts.push(p);
                jobs -= 1;
            }
            Err(_) => break,
        }
    }
    let mut hosts = crate::ssh::ssh_hosts();
    hosts.truncate(5);
    let ssh: Vec<Detected> = hosts
        .into_iter()
        .map(|h| {
            let mut c = cfg(&format!("Claude Code on {h}"), BackendKind::AgentCli, "claude-code");
            c.ssh_host = Some(h.clone());
            Detected { suggestion: c, reason: format!("SSH host {h} from ~/.ssh/config (runs an agent CLI there)"), needs_key: false }
        })
        .collect();
    parts.push((100, ssh));
    parts.sort_by_key(|(i, _)| *i);
    parts.into_iter().flat_map(|(_, v)| v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_detection() {
        let d = env_keys(|v| (v == "OPENROUTER_API_KEY").then(|| "k".to_string()));
        assert_eq!(d.len(), 4);
        assert_eq!(d[0].suggestion.kind, BackendKind::Openrouter);
        assert!(d[0].reason.contains("OPENROUTER_API_KEY"));
        // Kimi K3 rides on the same key, so it needs none of its own.
        let kimi = d.iter().find(|d| d.suggestion.name == KIMI_NAME).unwrap();
        assert_eq!((kimi.suggestion.model.as_str(), kimi.needs_key), (crate::openai::KIMI_K3, false));
        // Without a key, Kimi K3 is still offered, asking for one.
        let none = env_keys(|_| Some("  ".into()));
        assert_eq!(none.len(), 3);
        assert!(none.iter().all(|d| d.needs_key));
        for kind in [BackendKind::Gemini, BackendKind::Xai] {
            let found = env_keys(|_| Some("sk-test".into()));
            assert!(!found.iter().find(|d| d.suggestion.kind == kind).unwrap().needs_key);
        }
    }

    #[test]
    fn closed_port_is_fast_and_negative() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        drop(l);
        let t = Instant::now();
        assert!(probe_local("X", &format!("http://127.0.0.1:{port}/v1")).is_none());
        assert!(t.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn whole_scan_is_bounded() {
        let t = Instant::now();
        let _ = detect_all();
        assert!(t.elapsed() < TOTAL + Duration::from_millis(1500));
    }
}
