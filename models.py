"""
IncidentOps Environment Models.

Defines the typed Pydantic models for actions, observations, and state used
in the IncidentOps OpenEnv environment.

Action space:
    IncidentAction - A text command issued by the agent (e.g. "logs api-gateway")

Observation space:
    IncidentObservation - Rich terminal-style output plus structured metadata

State:
    Uses the base openenv State model (episode_id, step_count)
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation


class IncidentAction(Action):
    """
    An action issued by the agent to the incident response environment.

    The agent interacts via a text command interface that mirrors a real
    ops terminal. Each command targets a specific investigation or
    remediation operation.

    Attributes:
        command: The full command string, including arguments.
                 Examples:
                   "help"
                   "logs api-gateway"
                   "metrics payment-processor"
                   "restart payment-processor"
                   "config api-gateway pool_size 100"
                   "resolve"

    Available top-level commands:
        help                          -- List all available commands
        status                        -- System-wide status dashboard
        alerts                        -- View active alerts with severity
        logs <service>                -- Recent log entries for a service
        metrics <service>             -- CPU / mem / latency / error-rate
        trace <request_id>            -- Follow a request through services
        diagnose <service>            -- Deep diagnostic report for service
        restart <service>             -- Restart a service pod
        scale <service> <replicas>    -- Scale service replica count
        rollback <service>            -- Roll back last deployment
        failover <service>            -- Trigger standby failover
        config <service> <key> <val>  -- Update a live config value
        notify <channel> <message>    -- Post status update
        resolve                       -- Mark incident resolved (ends episode)
    """

    command: str


class IncidentObservation(Observation):
    """
    Observation returned after each action.

    Attributes:
        output:            Terminal-style text output from the command.
        timestamp:         Simulated wall-clock time (ISO-8601 string).
        alert_count:       Number of currently active alerts.
        severity:          Current overall incident severity.
                           One of: "critical", "high", "medium", "low", "none".
        affected_services: List of service names showing degradation.
        done:              True when the episode has ended.
        reward:            Scalar reward in open interval (0.01, 0.99).
        metadata:          Pass-through dict for additional info.
    """

    output: str
    timestamp: str
    alert_count: int = 0
    severity: str = "none"
    affected_services: List[str] = []
