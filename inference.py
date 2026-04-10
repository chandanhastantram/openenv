"""
IncidentOps Baseline Inference Script.

Runs a language model agent against all three IncidentOps tasks and produces
a reproducible baseline score. Emits structured log output to stdout.

This script is SELF-CONTAINED: it communicates with the environment server
via plain HTTP (urllib), so it works without the incident_ops_env package
being installed.

Environment variables (mandatory before submission):
    API_BASE_URL   — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     — Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       — Hugging Face token / API key

Optional:
    INCIDENT_BASE_URL — Running server URL (default: http://localhost:8000)

Stdout format (as required by OpenEnv):
    [START] task=<task_name> env=incident_ops_env model=<model>
    [STEP]  step=<n> action=<cmd> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import re
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
INCIDENT_BASE_URL: str = os.getenv("INCIDENT_BASE_URL", "http://localhost:8000")

ENV_NAME = "incident_ops_env"
TASKS = ["service-restart", "config-drift", "cascading-failure"]
MAX_STEPS = 20
TEMPERATURE = 0.1
MAX_TOKENS = 120
FALLBACK_ACTION = "status"

# ─────────────────────────────────────────────────────────────────────────────
#  Structured logging (required format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    done_str = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-HTTP environment client (no package import required)
# ─────────────────────────────────────────────────────────────────────────────

def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST JSON to *url* and return the parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


class _Obs:
    """Lightweight observation wrapper over the raw API dict."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        obs = raw.get("observation", raw)
        self.output: str = obs.get("output", "")
        self.timestamp: str = obs.get("timestamp", "")
        self.alert_count: int = int(obs.get("alert_count", 0))
        self.severity: str = obs.get("severity", "none")
        self.affected_services: List[str] = obs.get("affected_services", [])
        self.done: bool = bool(obs.get("done", False))
        # reward may live at top level or inside observation
        self.reward: float = float(raw.get("reward") or obs.get("reward") or 0.5)


class EnvHttpClient:
    """Talks to the IncidentOps server via HTTP without any package imports."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str) -> _Obs:
        resp = _http_post(f"{self.base_url}/reset", {"task_name": task_name})
        return _Obs(resp)

    def step(self, command: str) -> _Obs:
        resp = _http_post(f"{self.base_url}/step", {"command": command})
        return _Obs(resp)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM client (openai SDK — listed as a dependency so it will be installed)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(client: Any, messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return the response text, or '' on failure."""
    try:
        from openai import OpenAI  # imported here to give a clear error if missing
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a production incident.
You interact with a monitoring system through text commands.

Your goal is to:
1. Quickly triage the incident by checking alerts and system status
2. Identify the root cause through logs, metrics, and diagnostics
3. Apply the correct remediation (restart, rollback, failover, or config change)
4. Mark the incident as resolved with the 'resolve' command

Available commands:
  help                          -- Show all commands
  status                        -- System dashboard
  alerts                        -- Active alerts
  logs <service>                -- Service logs
  metrics <service>             -- Service metrics
  trace <request_id>            -- Trace a request
  diagnose <service>            -- Deep diagnostic
  restart <service>             -- Restart a service
  scale <service> <n>           -- Scale replicas
  rollback <service>            -- Rollback deployment
  failover <service>            -- Trigger DB failover
  config <service> <key> <val>  -- Update configuration
  notify <channel> <message>    -- Send status update
  resolve                       -- Mark incident resolved

RULES:
- Reply with EXACTLY ONE command per turn. No explanations.
- Do NOT restart healthy services.
- Start with 'alerts' then 'status' to orient yourself.
- Use 'diagnose <service>' on suspicious services.
- Type 'resolve' once you have applied the fix and verified recovery.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Action parser
# ─────────────────────────────────────────────────────────────────────────────

VALID_VERBS = {
    "help", "status", "alerts", "logs", "metrics", "trace",
    "diagnose", "restart", "scale", "rollback", "failover",
    "config", "notify", "resolve",
}

_PREAMBLE_RE = re.compile(r"^(action|next action|command)[:\-]\s*", re.IGNORECASE)


def parse_action(response_text: str) -> str:
    """Extract a clean command string from model output."""
    if not response_text:
        return FALLBACK_ACTION

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _PREAMBLE_RE.sub("", line).strip()
        parts = line.split()
        if parts and parts[0].lower() in VALID_VERBS:
            return line  # Return the whole line (verb + args)

    # Last-resort: return first non-empty line cleaned up
    first = response_text.strip().splitlines()[0].strip() if response_text.strip() else ""
    first = _PREAMBLE_RE.sub("", first).strip()
    return first if first else FALLBACK_ACTION


# ─────────────────────────────────────────────────────────────────────────────
#  Agent loop (one task)
# ─────────────────────────────────────────────────────────────────────────────

def run_task(llm_client: Any, env: EnvHttpClient, task_name: str) -> None:
    """Run one complete task episode and emit structured logs."""
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    history: List[str] = []

    try:
        # ── Reset ─────────────────────────────────────────────────────
        obs = env.reset(task_name=task_name)

        for step_idx in range(1, MAX_STEPS + 1):
            if obs.done:
                success = obs.reward > 0.5
                break

            # ── Build prompt ──────────────────────────────────────────
            history_text = "\n".join(history[-6:]) if history else "None"
            user_content = textwrap.dedent(f"""
                CURRENT OBSERVATION:
                {obs.output}

                System state:
                  Severity:          {obs.severity.upper()}
                  Active alerts:     {obs.alert_count}
                  Affected services: {', '.join(obs.affected_services) or 'none'}
                  Sim time:          {obs.timestamp}

                Recent actions:
                {history_text}

                Reply with EXACTLY ONE command to take next.
            """).strip()

            # ── Call LLM ─────────────────────────────────────────────
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            response_text = _call_llm(llm_client, messages)
            action_str = parse_action(response_text)

            # ── Step environment ──────────────────────────────────────
            try:
                obs = env.step(action_str)
            except Exception as exc:
                print(f"[DEBUG] env.step failed: {exc}", file=sys.stderr, flush=True)
                # Emit a mid-episode error step and abort this task
                log_step(step=step_idx, action=action_str, reward=0.5, done=True, error=str(exc))
                rewards.append(0.5)
                steps_taken = step_idx
                success = False
                return

            reward = float(obs.reward)
            # Clamp into the open interval (0, 1) as the validator requires
            reward = max(0.01, min(0.99, reward))
            done = obs.done

            rewards.append(reward)
            steps_taken = step_idx

            log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step_idx}: {action_str} -> reward {reward:.2f}")

            if done:
                success = reward > 0.5
                break

        else:
            # MAX_STEPS exhausted
            success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import OpenAI here so a missing install gives a clear message
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to the already-running environment server via plain HTTP
    env = EnvHttpClient(base_url=INCIDENT_BASE_URL)

    for task in TASKS:
        run_task(llm_client=llm_client, env=env, task_name=task)


if __name__ == "__main__":
    main()
