//! Bounded retries and circuit breaking shared by API transports.
use crate::model::{ChatModel, Delta, Stop, TurnRequest, TurnResult};
use craft_anything_evidence::{ErrorCode, ToolError};
use std::time::{Duration, Instant};

pub struct Gateway {
    inner: Box<dyn ChatModel>,
    failures: u32,
    open_until: Option<Instant>,
    secret: Option<String>,
}

impl Gateway {
    pub fn new(inner: Box<dyn ChatModel>) -> Self {
        Self { inner, failures: 0, open_until: None, secret: None }
    }
    pub fn with_secret(mut self, secret: Option<String>) -> Self {
        self.secret = secret.filter(|s| !s.is_empty());
        self
    }
}

fn transient(error: &ToolError) -> bool {
    matches!(error.code, ErrorCode::OverLimit | ErrorCode::EngineMissing | ErrorCode::TimedOut)
}

fn backoff(stop: &Stop, duration: Duration) -> Result<(), ToolError> {
    let until = Instant::now() + duration;
    while Instant::now() < until {
        stop.check()?;
        std::thread::sleep(Duration::from_millis(50).min(until.saturating_duration_since(Instant::now())));
    }
    stop.check()
}

impl ChatModel for Gateway {
    fn model(&self) -> String {
        self.inner.model()
    }
    fn turn(&mut self, req: &TurnRequest<'_>, on_delta: &mut dyn FnMut(Delta), stop: &Stop) -> Result<TurnResult, ToolError> {
        stop.check()?;
        if self.open_until.is_some_and(|until| Instant::now() < until) {
            return Err(ToolError::unavailable("This provider is cooling down after repeated failures.")
                .with_remediation("Wait 30 seconds or use another configured backend."));
        }
        for attempt in 0..3 {
            let mut produced = false;
            let result = self.inner.turn(
                req,
                &mut |delta| {
                    produced = true;
                    on_delta(delta);
                },
                stop,
            );
            match result {
                Ok(result) => {
                    self.failures = 0;
                    self.open_until = None;
                    return Ok(result);
                }
                Err(mut error) => {
                    if let Some(secret) = &self.secret {
                        error.message = error.message.replace(secret, "[redacted]");
                        error.remediation = error.remediation.map(|r| r.replace(secret, "[redacted]"));
                        if let Some(details) = error.details.as_mut() {
                            redact_value(details, secret, 0);
                        }
                    }
                    if error.code == ErrorCode::Cancelled {
                        return Err(error);
                    }
                    if transient(&error) {
                        self.failures = self.failures.saturating_add(1);
                        if self.failures >= 3 {
                            self.open_until = Some(Instant::now() + Duration::from_secs(30));
                        }
                    }
                    if produced || !transient(&error) || attempt == 2 || self.open_until.is_some_and(|until| Instant::now() < until) {
                        return Err(error);
                    }
                    backoff(stop, Duration::from_millis(250 * (1 << attempt)))?;
                }
            }
        }
        Err(ToolError::unavailable("The provider retry limit was reached."))
    }
}

fn redact_value(value: &mut serde_json::Value, secret: &str, depth: u32) {
    if depth >= 64 {
        *value = serde_json::Value::String("[redacted]".into());
        return;
    }
    match value {
        serde_json::Value::String(s) => *s = s.replace(secret, "[redacted]"),
        serde_json::Value::Array(values) => {
            for v in values {
                redact_value(v, secret, depth + 1);
            }
        }
        serde_json::Value::Object(values) => {
            for v in values.values_mut() {
                redact_value(v, secret, depth + 1);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Effort;
    struct Broken {
        calls: u32,
        stream: bool,
    }
    impl ChatModel for Broken {
        fn model(&self) -> String {
            "fake".into()
        }
        fn turn(&mut self, _: &TurnRequest<'_>, delta: &mut dyn FnMut(Delta), _: &Stop) -> Result<TurnResult, ToolError> {
            self.calls += 1;
            if self.stream {
                delta(Delta::Text("partial".into()));
            }
            Err(ToolError::new(ErrorCode::OverLimit, "rate limited"))
        }
    }
    #[test]
    fn partial_output_is_never_retried_and_cancellation_is_immediate() {
        let req = TurnRequest { system: "", messages: &[], tools: &[], effort: Effort::Low, max_tokens: 10 };
        let mut gateway = Gateway::new(Box::new(Broken { calls: 0, stream: true }));
        let mut deltas = 0;
        assert!(gateway.turn(&req, &mut |_| deltas += 1, &Stop::default()).is_err());
        assert_eq!(deltas, 1);
        let stop = Stop::default();
        stop.stop();
        assert_eq!(gateway.turn(&req, &mut |_| {}, &stop).unwrap_err().code, ErrorCode::Cancelled);
    }
    #[test]
    fn repeated_failures_open_the_circuit() {
        let req = TurnRequest { system: "", messages: &[], tools: &[], effort: Effort::Low, max_tokens: 10 };
        let mut gateway = Gateway::new(Box::new(Broken { calls: 0, stream: false }));
        assert!(gateway.turn(&req, &mut |_| {}, &Stop::default()).is_err());
        assert_eq!(gateway.failures, 3);
        assert!(gateway.open_until.is_some());
        assert!(gateway.turn(&req, &mut |_| {}, &Stop::default()).unwrap_err().message.contains("cooling down"));
    }
    #[test]
    fn transport_errors_cannot_echo_the_api_key() {
        struct Leaky;
        impl ChatModel for Leaky {
            fn model(&self) -> String {
                "fake".into()
            }
            fn turn(&mut self, _: &TurnRequest<'_>, _: &mut dyn FnMut(Delta), _: &Stop) -> Result<TurnResult, ToolError> {
                Err(ToolError::new(ErrorCode::NoPermission, "Rejected sk-test-auth")
                    .with_remediation("Replace sk-test-auth")
                    .with_details(serde_json::json!({"nested":["sk-test-auth"]})))
            }
        }
        let mut gateway = Gateway::new(Box::new(Leaky)).with_secret(Some("sk-test-auth".into()));
        let req = TurnRequest { system: "", messages: &[], tools: &[], effort: Effort::Low, max_tokens: 10 };
        let error = gateway.turn(&req, &mut |_| {}, &Stop::default()).unwrap_err();
        assert!(!error.message.contains("sk-test-auth"));
        assert!(!error.remediation.unwrap().contains("sk-test-auth"));
        assert!(!error.details.unwrap().to_string().contains("sk-test-auth"));
    }
}
