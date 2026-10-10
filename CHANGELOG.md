# Changelog

## 0.3.0

- `google-adk` is no longer an install dependency. It is a peer dependency: install it in your agent project (`google-adk>=1.39.1,<2`), or use the `adk` extra: `pip install "resilient-gemini[adk]"`. Importing `resilient_gemini` without it raises a clear `ImportError`. For development it is in the `dev` group, so `uv sync` still installs it.

## 0.2.0

- Opt-in Priority PayGo via `RESILIENT_GEMINI_PRIORITY_PAYGO=priority|spillover` (env var only, off by default).
- Logs a startup WARNING when enabled, a per-request line noting the ~2x cost, and the served `traffic_type` (WARNING if downgraded to Standard PayGo).
- uv project layout (`uv sync`, `uv run pytest`, `uv build`).

## 0.1.0

- Linear retries on the primary model (60, 120, 180, 240 s), then fallback to a backup model.
- Per-model thinking level.
- Settings via keyword arguments or `RESILIENT_GEMINI_*` env vars.
