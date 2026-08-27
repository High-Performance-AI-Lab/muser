use minijinja::value::{from_args, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value as JinjaValue};
use serde_json::Value;

pub fn render(
    template_source: &str,
    messages: &Value,
    tools: Option<&Value>,
    current_date: &str,
) -> Result<String, Error> {
    render_with_options(template_source, messages, tools, current_date, true)
}

pub fn render_with_options(
    template_source: &str,
    messages: &Value,
    tools: Option<&Value>,
    current_date: &str,
    add_generation_prompt: bool,
) -> Result<String, Error> {
    let mut environment = Environment::new();
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    environment.add_function("strftime_now", {
        let current_date = current_date.to_owned();
        move |format: String| -> Result<String, Error> {
            if format == "%Y-%m-%d" {
                Ok(current_date.clone())
            } else {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "only the pinned %Y-%m-%d clock format is supported",
                ))
            }
        }
    });
    // Hugging Face templates execute with Python string and mapping methods.
    // MiniJinja intentionally does not expose those methods by default; the
    // pinned Muse template uses `str.split`, `map.get`, and `map.items`.
    // Implement those exact primitives instead of rewriting immutable GGUF
    // template bytes.
    environment.set_unknown_method_callback(|_state, value, method, args| {
        match (value.kind(), method) {
            (ValueKind::String, "split") => {
                let Some(text) = value.as_str() else {
                    return Err(Error::from(ErrorKind::UnknownMethod));
                };
                let (separator, max_splits): (Option<&str>, Option<usize>) = from_args(args)?;
                let pieces = match separator {
                    Some(separator) => text
                        .splitn(
                            max_splits.map_or(usize::MAX, |count| count.saturating_add(1)),
                            separator,
                        )
                        .map(JinjaValue::from)
                        .collect::<Vec<_>>(),
                    None => text
                        .split_whitespace()
                        .map(JinjaValue::from)
                        .collect::<Vec<_>>(),
                };
                Ok(JinjaValue::from(pieces))
            }
            (ValueKind::Map, "get") => {
                let Some(map) = value.as_object() else {
                    return Err(Error::from(ErrorKind::UnknownMethod));
                };
                let (key, default): (&JinjaValue, Option<JinjaValue>) = from_args(args)?;
                Ok(map
                    .get_value(key)
                    .unwrap_or_else(|| default.unwrap_or_else(|| JinjaValue::from(()))))
            }
            (ValueKind::Map, "items") => {
                let Some(map) = value.as_object() else {
                    return Err(Error::from(ErrorKind::UnknownMethod));
                };
                let () = from_args(args)?;
                Ok(JinjaValue::make_object_iterable(
                    map.clone(),
                    |map| match map.try_iter_pairs() {
                        Some(iter) => {
                            Box::new(iter.map(|(key, value)| JinjaValue::from(vec![key, value])))
                        }
                        None => Box::new(None.into_iter()),
                    },
                ))
            }
            _ => Err(Error::from(ErrorKind::UnknownMethod)),
        }
    });
    // MiniJinja's built-in `tojson` canonicalizes map keys. The pinned HF
    // renderer serializes ordered JSON objects in their source order, which
    // is part of `/apply-template` byte identity and materially affects the
    // model's tool-schema prompt. Keep the original serde map order here.
    environment.add_filter(
        "muser_tojson",
        |value: JinjaValue| -> Result<JinjaValue, Error> {
            let json = serde_json::to_string(&value).map_err(|error| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    "cannot serialize template JSON",
                )
                .with_source(error)
            })?;
            Ok(JinjaValue::from_safe_string(python_json_spacing(&json)))
        },
    );
    // MiniJinja names the Jinja/HF `iterable` test `sequence`; this is a
    // parser-dialect alias only and does not change rendered bytes.
    let compatible = template_source
        .replace(" is iterable", " is sequence")
        .replace("| tojson", "| muser_tojson")
        .replace("|tojson", "|muser_tojson")
        .replace(
            "namespace(name=tcid if tcid else '')",
            "namespace(name=(tcid or ''))",
        )
        .replace("-%}", "-%}\n")
        .replace("-}}", "-}}\n");
    environment.add_template("muse-onyx-atem", &compatible)?;
    let rendered = environment
        .get_template("muse-onyx-atem")?
        .render(minijinja::context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => add_generation_prompt,
            bos_token => "",
            current_date => current_date,
            reasoning_strength => "high",
        })?;
    Ok(tools.map_or(rendered.clone(), |tools| {
        restore_tool_schema_order(&rendered, tools)
    }))
}

/// OpenAI-compatible requests carry `tool_calls[*].function.arguments` as a
/// JSON string, while the pinned Muse template requires a mapping (it iterates
/// `args.items()` and raises on non-mappings). Parse object-valued JSON
/// strings into mappings for template evaluation only. Wire types and stored
/// messages are untouched, and strings that do not hold a JSON object are
/// left as-is so malformed input still fails closed inside the template.
pub fn normalize_tool_call_arguments(message: &mut Value) {
    let Some(tool_calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
        return;
    };
    for tool_call in tool_calls {
        let Some(arguments) = tool_call
            .get_mut("function")
            .and_then(|function| function.get_mut("arguments"))
        else {
            continue;
        };
        let Some(raw) = arguments.as_str() else {
            continue;
        };
        if let Ok(parsed @ Value::Object(_)) = serde_json::from_str::<Value>(raw) {
            *arguments = parsed;
        }
    }
}

fn python_json_spacing(compact: &str) -> String {
    let mut output = String::with_capacity(compact.len() + compact.len() / 8);
    let mut in_string = false;
    let mut escaped = false;
    for character in compact.chars() {
        output.push(character);
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
        } else if character == '"' {
            in_string = true;
        } else if matches!(character, ',' | ':') {
            output.push(' ');
        }
    }
    output
}

fn restore_tool_schema_order(rendered: &str, tools: &Value) -> String {
    #[derive(serde::Serialize)]
    struct OrderedToolSchema<'a> {
        name: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<&'a Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<&'a Value>,
    }

    let replacements = tools
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("function"))
        .filter_map(|function| {
            let name = function.get("name")?.as_str()?;
            let ordered = OrderedToolSchema {
                name,
                description: function.get("description"),
                parameters: function.get("parameters"),
            };
            let compact = serde_json::to_string(&ordered).ok()?;
            Some((name.to_owned(), python_json_spacing(&compact)))
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut output = String::with_capacity(rendered.len());
    for line in rendered.split_inclusive('\n') {
        let (body, newline) = line
            .strip_suffix('\n')
            .map_or((line, ""), |body| (body, "\n"));
        let replacement = serde_json::from_str::<Value>(body)
            .ok()
            .filter(|value| value.get("parameters").is_some())
            .and_then(|value| value.get("name").and_then(Value::as_str).map(str::to_owned))
            .and_then(|name| replacements.get(&name));
        output.push_str(replacement.map_or(body, String::as_str));
        output.push_str(newline);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "release-real-model")]
    fn release_gguf() -> muser_engine::gguf::GgufFile {
        const MODEL_SHA256: &str =
            "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8";
        let declared = std::env::var("MUSER_MODEL_SHA256")
            .expect("release-real-model requires MUSER_MODEL_SHA256");
        assert_eq!(declared, MODEL_SHA256, "unknown release-model identity");
        let path = std::path::PathBuf::from(
            std::env::var("MUSER_MODEL").expect("release-real-model requires MUSER_MODEL"),
        );
        let metadata = std::fs::symlink_metadata(&path).expect("release GGUF must exist");
        assert!(
            metadata.file_type().is_file(),
            "release GGUF must be a regular file"
        );
        assert_eq!(metadata.len(), 16_756_681_056, "release GGUF byte size");
        let gguf = muser_engine::gguf::GgufFile::parse_path(&path).expect("pinned GGUF metadata");
        let template = gguf.chat_template().expect("pinned chat template");
        assert_eq!(template.len(), 7_167, "pinned chat-template byte length");
        assert_eq!(
            gguf.chat_template_sha256().map(hex32).as_deref(),
            Some("114f55ebdc1804c1af371197b9fdf2d6bb925966c9dfe46b73782a71bc07965e")
        );
        assert_eq!(
            hex32(gguf.tokenizer_metadata_sha256()),
            "61e73226502f8f54455555990c0000852247bbec32b107730ec544bc0b738055"
        );
        gguf
    }

    #[cfg(feature = "release-real-model")]
    fn hex32(value: [u8; 32]) -> String {
        value.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn renders_with_a_deterministic_clock() {
        let rendered = render(
            "{{ current_date }}:{{ messages[0].content }}",
            &serde_json::json!([{"role":"user", "content":"hi"}]),
            None,
            "2026-08-14",
        )
        .unwrap();
        assert_eq!(rendered, "2026-08-14:hi");
    }

    #[test]
    fn ordered_template_json_uses_python_separators() {
        assert_eq!(
            python_json_spacing(r#"{"a":"x,y:z","nested":{"b":1,"c":false}}"#),
            r#"{"a": "x,y:z", "nested": {"b": 1, "c": false}}"#
        );
    }

    #[test]
    fn python_mapping_methods_render_followup_tool_messages() {
        let rendered = render(
            "{{ messages[0].get('missing', 'fallback') }}|{% for key, value in messages[0].items() %}{{ key }}={{ value }};{% endfor %}",
            &serde_json::json!([{"name":"search","tool_call_id":"call-1"}]),
            None,
            "2026-08-17",
        )
        .unwrap();
        assert_eq!(rendered, "fallback|name=search;tool_call_id=call-1;");
    }

    #[test]
    fn stringified_tool_call_arguments_are_normalized_for_template_evaluation() {
        // Mirrors the pinned template's render_atem contract: arguments must
        // be a mapping or the template raises.
        let template = "{% for tc in messages[0].tool_calls %}{% set args = tc.function.arguments %}{% if args is not mapping %}{{ raise_exception('arguments must be a mapping') }}{% endif %}{% for key, value in args.items() %}{{ key }}={{ value }};{% endfor %}{% endfor %}";
        let mut message = serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "fs.read", "arguments": "{\"path\": \"README.md\"}"}
            }]
        });
        // The OpenAI string representation fails closed without normalization.
        assert!(render(
            template,
            &serde_json::json!([message.clone()]),
            None,
            "2026-08-17"
        )
        .is_err());
        normalize_tool_call_arguments(&mut message);
        let rendered = render(template, &serde_json::json!([message]), None, "2026-08-17").unwrap();
        assert_eq!(rendered, "path=README.md;");
    }

    #[test]
    fn non_object_tool_call_arguments_are_left_to_fail_closed() {
        let mut message = serde_json::json!({
            "tool_calls": [{"function": {"name": "broken", "arguments": "not json"}}]
        });
        normalize_tool_call_arguments(&mut message);
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("not json")
        );
        // A JSON string that does not hold an object is preserved as-is.
        let mut scalar = serde_json::json!({
            "tool_calls": [{"function": {"name": "broken", "arguments": "\"just a string\""}}]
        });
        normalize_tool_call_arguments(&mut scalar);
        assert_eq!(
            scalar["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("\"just a string\"")
        );
    }

    #[cfg(feature = "release-real-model")]
    #[test]
    fn real_gguf_renders_followup_tool_call_with_stringified_arguments() {
        let gguf = release_gguf();
        let tools = serde_json::json!([{
            "type":"function",
            "function": {
                "name":"clock.now",
                "description":"Read clock",
                "parameters": {"type":"object", "properties": {}, "additionalProperties": false}
            }
        }]);
        let mut assistant = serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "clock.now", "arguments": "{}"}
            }]
        });
        normalize_tool_call_arguments(&mut assistant);
        let messages = serde_json::json!([
            {"role":"user", "content":"What time is it?"},
            assistant,
            {"role":"tool", "tool_call_id":"call-1", "name":"clock.now", "content":"{\"time\": \"noon\"}"}
        ]);
        let rendered = render(
            gguf.chat_template().expect("chat template"),
            &messages,
            Some(&tools),
            "2026-08-14",
        )
        .expect("render follow-up tool-call turn with stringified arguments");
        assert!(rendered.ends_with("<|start|>assistant"), "{rendered}");
    }

    #[cfg(feature = "release-real-model")]
    #[test]
    fn real_gguf_tool_call_turn_drops_assistant_content() {
        // Template contract that build_prefill relies on: an assistant
        // message carrying tool_calls renders ATEM blocks only; its content
        // is not part of the rendered prompt.
        let gguf = release_gguf();
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": "prose the template must drop",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "clock.now", "arguments": {"tz": "utc"}}
            }]
        });
        let messages = serde_json::json!([
            {"role":"user", "content":"What time is it?"},
            assistant,
            {"role":"tool", "tool_call_id":"call-1", "name":"clock.now", "content":"{\"time\": \"noon\"}"}
        ]);
        let rendered = render(
            gguf.chat_template().expect("chat template"),
            &messages,
            None,
            "2026-08-14",
        )
        .expect("render tool-call turn with assistant content");
        assert!(
            !rendered.contains("prose the template must drop"),
            "{rendered}"
        );
        assert!(
            rendered.contains("<atem:parameter name=\"tz\">utc</atem:parameter>"),
            "{rendered}"
        );
        assert!(
            rendered.contains("<tool_output name=\"clock.now\">"),
            "{rendered}"
        );
    }

    #[cfg(feature = "release-real-model")]
    #[test]
    fn real_gguf_basic_prompt_matches_pinned_llama() {
        let gguf = release_gguf();
        let rendered = render(
            gguf.chat_template().expect("chat template"),
            &serde_json::json!([{"role":"user", "content":"hi"}]),
            None,
            "2026-08-14",
        )
        .expect("render pinned template");
        assert_eq!(
            rendered,
            "<|start|>system<|message|>You are a helpful AI assistant.\nKnowledge cutoff: 2026-01-04.\nCurrent date: 2026-08-14.\n\nReasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|><|start|>user<|message|>hi<|eot|><|start|>assistant"
        );
    }

    #[cfg(feature = "release-real-model")]
    #[test]
    fn real_gguf_tool_prompt_uses_python_split_semantics() {
        let gguf = release_gguf();
        let rendered = render(
            gguf.chat_template().expect("chat template"),
            &serde_json::json!([{"role":"user", "content":"What time is it?"}]),
            Some(&serde_json::json!([{
                "type":"function",
                "function": {
                    "name":"clock.now",
                    "description":"Read clock",
                    "parameters": {"type":"object", "properties": {}, "additionalProperties": false}
                }
            }])),
            "2026-08-14",
        )
        .expect("render pinned tool template");
        assert!(rendered.contains("<atem:invoke name=\"$FUNCTION_NAME\">"));
        assert!(rendered.contains(
            r#"{"name": "clock.now", "description": "Read clock", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}"#
        ), "{rendered}");
        assert!(rendered.ends_with("<|start|>assistant"));
    }
}
