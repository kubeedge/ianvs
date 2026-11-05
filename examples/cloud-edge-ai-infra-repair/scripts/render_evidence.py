# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Render actual local evidence for documentation screenshots (not a terminal)."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra_repair.io_utils import save_json

STYLE = """
body{margin:0;background:#eef2f6;color:#162a43;font:17px/1.5 Arial,sans-serif}
main{max-width:1180px;margin:32px auto;padding:32px;background:white;border-radius:12px}
h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin-top:24px}
.note{color:#53677d}.tag{display:inline-block;padding:5px 12px;background:#e2edf7;border-radius:5px;margin:4px}
table{border-collapse:collapse;width:100%;margin:16px 0}td,th{text-align:left;border-bottom:1px solid #dce3ea;padding:9px}
pre{white-space:pre-wrap;word-break:break-word;background:#122033;color:#e7eef7;padding:18px;border-radius:8px;font:14px/1.45 Consolas,monospace}
small{display:block;color:#53677d;word-break:break-all}a{color:#17629d}
"""


def page(title, body):
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title) +
            '</title><style>' + STYLE + '</style><main><h1>' + html.escape(title) +
            '</h1><p class="note">Actual WSL experiment evidence • rendered report, not a native terminal capture</p>' +
            body + '</main></html>')


def table(values):
    def display(value):
        if value is None:
            return "Not measured / not exercised"
        if isinstance(value, float):
            return "%.3f" % value
        return str(value)
    return '<table>' + ''.join('<tr><th>' + html.escape(str(key)) + '</th><td>' + html.escape(display(value)) +
                              '</td></tr>' for key, value in values.items()) + '</table>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="workspace/infra-repair")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    sources, links = [], []
    index_path = output / "experiment-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    def record_source(path):
        sources.append({"path": str(path.relative_to(root))})

    preflight = root / "preflight.json"
    if preflight.exists():
        data = json.loads(preflight.read_text(encoding="utf-8"))
        record_source(preflight)
        body = table({"OS": data["platform"], "Python": data["python"].splitlines()[0],
                      **data["packages"]})
        body += '<h2>GPU / Docker observations</h2><pre>' + html.escape(json.dumps(data["commands"], indent=2)) + '</pre>'
        (output / "00-environment.html").write_text(page("ENV-001 | WSL environment preflight", body), encoding="utf-8")
        links.append("00-environment.html")
    paths = sorted(root.glob("batch-*/*/result.json"),
                   key=lambda path: json.loads(path.read_text(encoding="utf-8"))["started_at"])
    cluster = root / "cluster-environment.json"
    if cluster.exists():
        record_source(cluster)
        data = json.loads(cluster.read_text(encoding="utf-8"))
        body = table({key: data[key] for key in ("captured_at", "topology", "source")})
        for node in data["nodes"]:
            body += table(node)
        (output / "03-cluster.html").write_text(page("ENV-002 | Dedicated KubeEdge cluster", body), encoding="utf-8")
        links.append("03-cluster.html")
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["run_id"] not in index:
            index[row["run_id"]] = {"experiment": "EXP-%03d" % (len(index) + 1),
                                     "result": str(path.relative_to(root))}
        experiment = index[row["run_id"]]["experiment"]
        record_source(path)
        body = '<span class="tag">' + html.escape(row["scenario"]) + '</span><span class="tag">' + html.escape(row["agent"]) + '</span>'
        contract = path.parent / "agent-contract.json"
        if contract.exists():
            record_source(contract)
            body += '<span class="tag">' + html.escape(json.loads(contract.read_text(encoding="utf-8"))["revision"]) + '</span>'
        body += table({"experiment": experiment, **{key: row.get(key) for key in ("started_at", "status", "injection_success",
                      "repair_success", "root_cause_correct", "repair_seconds", "rollback_success", "cleanup_success",
                      "tool_calls", "denied_calls", "unauthorized_executions")}})
        verification = row.get("verification") or {}
        body += table({key: verification.get(key) for key in ("correctness_pass", "p95_latency_seconds", "p95_ttft_seconds", "slo_pass")})
        body += '<p class="note">Single-run evidence; full acceptance requires complete reference and rollback cohorts. Null = not measured or not exercised.</p>'
        if row.get("errors"):
            body += '<h2>Recorded errors</h2><pre>' + html.escape(json.dumps(row["errors"], indent=2)) + '</pre>'
        if row.get("diagnosis"):
            body += '<h2>Agent diagnosis (independently scored above)</h2>' + table(row["diagnosis"])
        trace = path.parent / "agent.jsonl"
        if trace.exists():
            record_source(trace)
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
            actions = []
            for event in events:
                try:
                    action = json.loads(event["response"])
                    actions.append(str(event["step"] + 1) + '. ' + str(action.get("tool", action.get("action", action.get("cause", "invalid")))))
                except (ValueError, AttributeError):
                    actions.append(str(event["step"] + 1) + '. invalid JSON action')
            body += '<h2>Recorded agent actions</h2><pre>' + html.escape('\n'.join(actions)) + '</pre>'
        fault = path.parent / "fault.log"
        if fault.exists():
            record_source(fault)
            text = re.sub(r"\x1b\[[0-9;]*m", "", fault.read_text(encoding="utf-8", errors="replace"))
            text = text.replace(str(path.parent), "<" + experiment + " workspace>")
            text = re.sub(r"[^\s\"']*batch-[0-9a-f]{32}/[0-9a-f]{32}",
                          "<" + experiment + " workspace>", text)
            body += '<h2>Fault evidence (last 12 original log lines)</h2><pre>' + html.escape('\n'.join(text.splitlines()[-12:])) + '</pre>'
            body += '<small>Long workspace prefix shortened to the experiment number for readability; original log retained.</small>'
        body += '<small>Original result and log files are mapped in experiment-index.json under ' + experiment + '.</small>'
        filename = experiment + '.html'
        (output / filename).write_text(page(experiment + " | AI infrastructure repair experiment", body), encoding="utf-8")
        links.append(filename)
    agent_log = root / "local-agent/server.log"
    if agent_log.exists():
        record_source(agent_log)
        text = re.sub(r"\x1b\[[0-9;]*m", "", agent_log.read_text(encoding="utf-8", errors="replace"))
        body = ""
        smoke = root / "local-agent/chat-smoke.json"
        if smoke.exists():
            record_source(smoke)
            response = json.loads(smoke.read_text(encoding="utf-8"))
            body += table({"experiment": "DEPLOY-001", "model": response["model"],
                           "chat_response": response["choices"][0]["message"]["content"],
                           "tokens": response["usage"]["total_tokens"]})
        body += '<pre>' + html.escape('\n'.join(text.splitlines()[-25:])) + '</pre>'
        body += '<small>Source: local-agent/server.log • Screenshot records this render time only.</small>'
        (output / "01-vllm-server.html").write_text(page("DEPLOY-001 | Local reference agent — vLLM", body), encoding="utf-8")
        links.append("01-vllm-server.html")
    smoke = root / "local-agent-1.5b/chat-smoke.json"
    if smoke.exists():
        record_source(smoke)
        response = json.loads(smoke.read_text(encoding="utf-8"))
        body = table({"experiment": "DEPLOY-002", "model": response["model"],
                      "endpoint": response["base_url"], "chat_response": response["generated_text"],
                      "tokens": response["usage"]["total_tokens"]})
        body += '<p>Actual generated response from the independent local agent service; readiness alone is not benchmark success.</p>'
        (output / "02-vllm-1.5b.html").write_text(page("DEPLOY-002 | Qwen2.5-1.5B-Instruct", body), encoding="utf-8")
        links.append("02-vllm-1.5b.html")
    assessment = root / "acceptance.json"
    if assessment.exists():
        record_source(assessment)
        result = json.loads(assessment.read_text(encoding="utf-8"))
        body = table({"passed": result["passed"], "scope": result["scope"]})
        summary = result["reference_summary"]
        body += '<h2>Reference agent</h2>' + table({key: summary.get(key) for key in (
            "runs", "repair_rate", "root_cause_accuracy", "mean_success_repair_seconds",
            "tool_calls", "denied_calls", "unauthorized_executions")})
        rollback = result["rollback_summary"]
        body += '<h2>Independent rollback cohort</h2>' + table({key: rollback.get(key) for key in (
            "runs", "injection_rate", "cleanup_rate", "rollback_rate", "rollback_exercises")})
        body += '<h2>Reference results by scenario</h2><table><tr><th>Scenario</th><th>Runs</th><th>Repair rate</th><th>Root-cause accuracy</th></tr>'
        for scenario, group in summary.get("per_scenario", {}).items():
            body += '<tr><td>' + html.escape(scenario) + '</td><td>' + str(group["runs"]) + '</td><td>' + format(group["repair_rate"], ".1%") + '</td><td>' + format(group["root_cause_accuracy"], ".1%") + '</td></tr>'
        body += '</table><h2>Acceptance checks</h2>'
        body += table({"passed_checks": sum(result["checks"].values()), "total_checks": len(result["checks"])})
        body += '<details><summary>Individual checks (also retained in acceptance.json)</summary>' + table(result["checks"]) + '</details>'
        (output / "acceptance.html").write_text(page("ACCEPT-001 | Complete cohort assessment", body), encoding="utf-8")
        links.append("acceptance.html")
    save_json(output / "sources.json", {"rendered_at": datetime.now(timezone.utc).isoformat(), "sources": sources})
    save_json(index_path, index)
    body = '<p>Rendered at ' + datetime.now(timezone.utc).isoformat() + '</p><ul>'
    body += ''.join('<li><a href="' + html.escape(link) + '">' + html.escape(link) + '</a></li>' for link in links) + '</ul>'
    (output / "index.html").write_text(page("WSL experiment evidence index", body), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
