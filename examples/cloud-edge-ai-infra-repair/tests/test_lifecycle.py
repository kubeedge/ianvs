# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Harness contract tests; fake services are never benchmark evidence."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import sys
import tempfile
import time
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra_repair.agents import CalibrationController, FailureController, LocalAgent
from infra_repair.acceptance import assess
from infra_repair.triage_agent import TriageAgent
from infra_repair.diagnosis import score_diagnosis
from infra_repair.gateway import Gateway
from infra_repair.edge_backend import EdgeBackend
from infra_repair.io_utils import http_json, is_loopback_url, stream_tokens
from infra_repair.report import export, summarize
from infra_repair.runner import RunConfig, run_once


class FakeBackend:
    """In-memory lifecycle fixture, only used in unit tests."""

    inject_ok = True
    rollback_ok = True
    cleanup_ok = True
    prepare_error = False
    wrong_output = False
    cleanup_calls = 0

    def __init__(self, *args):
        self.manifest = {"model": "hash"}
        self.fault = False

    def prepare(self):
        if self.prepare_error:
            raise RuntimeError("prepare failed")

    def inject(self, scenario):
        self.fault = True
        return self.inject_ok

    def integrity(self):
        return {"model": True}

    def logs(self):
        return "Configured model directory does not exist"

    def inspect(self):
        return {"running": not self.fault}

    def repair(self, action):
        self.fault = False
        return {"started": True}

    def generate(self, prompt, timeout):
        if self.fault:
            raise RuntimeError("service unavailable")
        time.sleep(.001)
        return {"token_ids": [2 if self.wrong_output else 1]}

    def rollback(self):
        self.fault = False
        return self.rollback_ok

    def cleanup(self):
        type(self).cleanup_calls += 1
        return self.cleanup_ok


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = RunConfig("unused", str(self.root), max_latency_ratio=100)

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, backend=FakeBackend, agent=None):
        return run_once("S1", self.config, agent or CalibrationController(), backend)

    def test_success_and_cleanup(self):
        result = self.run_case()
        self.assertTrue(result["repair_success"])
        self.assertTrue(result["cleanup_success"])
        self.assertIsNone(result["rollback_success"])
        self.assertFalse(result["acceptance_eligible"])
        self.assertIsNone(result["verification"]["p95_ttft_seconds"])

    def test_failure_rolls_back(self):
        result = self.run_case(agent=FailureController())
        self.assertFalse(result["repair_success"])
        self.assertTrue(result["rollback_success"])
        self.assertTrue(result["safe_to_continue"])

    def test_invalid_injection_is_not_agent_failure(self):
        class InvalidInjection(FakeBackend):
            inject_ok = False
        result = self.run_case(InvalidInjection)
        self.assertEqual(result["status"], "environment_error")
        self.assertFalse(result["injection_success"])
        self.assertTrue(result["rollback_success"])

    def test_prepare_failure_still_cleans(self):
        class PrepareFailure(FakeBackend):
            prepare_error = True
        result = self.run_case(PrepareFailure)
        self.assertTrue(result["cleanup_success"])
        self.assertEqual(result["status"], "environment_error")

    def test_rollback_failure_stops_suite(self):
        class BadRollback(FakeBackend):
            rollback_ok = False
        result = self.run_case(BadRollback, FailureController())
        self.assertFalse(result["safe_to_continue"])
        self.assertFalse(result["rollback_success"])

    def test_cleanup_failure_stops_suite(self):
        class BadCleanup(FakeBackend):
            cleanup_ok = False
        self.assertFalse(self.run_case(BadCleanup)["safe_to_continue"])

    def test_interrupt_still_rolls_back_and_cleans(self):
        class Interrupted(FailureController):
            def run(self, gateway, deadline):
                raise KeyboardInterrupt()
        result = self.run_case(agent=Interrupted())
        self.assertTrue(result["rollback_success"])
        self.assertTrue(result["cleanup_success"])
        self.assertEqual(result["status"], "interrupted")
        self.assertFalse(result["safe_to_continue"])

    def test_wrong_output_is_not_repair_success(self):
        class ChangedOutput(FakeBackend):
            def repair(self, action):
                self.fault = False
                self.wrong_output = True
            def rollback(self):
                self.wrong_output = False
                return super().rollback()
        result = self.run_case(ChangedOutput)
        self.assertFalse(result["repair_success"])
        self.assertTrue(result["rollback_success"])

    def test_gateway_rejects_extra_args_and_bad_resource(self):
        gateway = Gateway(FakeBackend(), self.root / "audit.jsonl")
        for args in ({"resource": "production"}, {"resource": "target", "path": "/etc"}, None):
            self.assertIn("error", gateway.call("set_model_path", args))
        self.assertIn("error", gateway.call(["invalid"], {"resource": "target"}))
        self.assertEqual(gateway.denied, 4)
        self.assertEqual(gateway.executed_unauthorized, 0)
        self.assertEqual(len((self.root / "audit.jsonl").read_text().splitlines()), 4)

    def test_gateway_deadline_and_call_budget(self):
        gateway = Gateway(FakeBackend(), self.root / "audit.jsonl", max_calls=1)
        gateway.call("inspect_service", {"resource": "target"})
        self.assertIn("error", gateway.call("set_model_path", {"resource": "target"}))
        expired = Gateway(FakeBackend(), self.root / "audit.jsonl", deadline=time.monotonic() - 1)
        self.assertIn("error", expired.call("set_model_path", {"resource": "target"}))

    def test_reports_include_failed_attempts_and_nulls(self):
        records = [self.run_case(), self.run_case(agent=FailureController())]
        summary = export(records, self.root / "reports")
        self.assertEqual(summary["repair_rate"], .5)
        self.assertEqual(summary["rollback_exercises"], 1)
        self.assertTrue((self.root / "reports/results.csv").is_file())
        saved = json.loads((self.root / "reports/results.json").read_text())
        self.assertIsNone(saved[1]["repair_seconds"])

    def test_unimplemented_scenario_refuses_to_fake(self):
        with self.assertRaises(NotImplementedError):
            run_once("S5", self.config, CalibrationController(), FakeBackend)

    def test_invalid_config_rejected_before_run(self):
        self.config.budget_seconds = float("nan")
        with self.assertRaises(ValueError):
            self.run_case()

    def test_agent_endpoint_must_be_loopback(self):
        with self.assertRaises(ValueError):
            LocalAgent("http://example.com/v1", "model")

    def test_local_agent_uses_schema_and_records_contract(self):
        agent = LocalAgent("http://127.0.1.1:8001/v1", "test-model")
        gateway = Gateway(FakeBackend(), self.root / "audit.jsonl")
        actions = [{"tool": "inspect_service", "arguments": {"resource": "target"},
                    "cause": "unknown", "evidence": ""},
                   {"tool": "finish", "arguments": {"resource": "target"},
                    "cause": "config.model_path", "evidence": "Observed startup error"}]
        responses = [{"choices": [{"message": {"content": json.dumps(action)}}],
                      "usage": {"total_tokens": 7}} for action in actions]
        with patch("infra_repair.agents.http_json", side_effect=responses) as request:
            diagnosis = agent.run(gateway, time.monotonic() + 10)
        self.assertEqual(diagnosis["tokens"], 14)
        self.assertEqual(gateway.calls, 1)
        self.assertEqual(request.call_args.args[1]["response_format"]["type"], "json_schema")
        self.assertTrue((self.root / "agent-contract.json").is_file())

    def test_agent_failure_keeps_consumed_tokens(self):
        agent = LocalAgent("http://127.0.1.1:8001/v1", "test-model")
        gateway = Gateway(FakeBackend(), self.root / "audit.jsonl")
        response = {"choices": [{"message": {"content": json.dumps({
            "tool": "inspect_service", "arguments": {"resource": "target"}})}}],
            "usage": {"total_tokens": 17}}
        with patch("infra_repair.agents.http_json", side_effect=[response, TimeoutError("API timeout")]):
            with self.assertRaises(TimeoutError):
                agent.run(gateway, time.monotonic() + 10)
        self.assertEqual(agent.last_tokens, 17)

    def test_empty_report_not_ranked(self):
        with self.assertRaises(ValueError):
            summarize([])

    def test_triage_uses_model_diagnosis_and_cites_observed_evidence(self):
        agent = TriageAgent("http://127.0.1.1:8001/v1", "test-model")
        response = {"choices": [{"message": {"content": json.dumps({
            "cause": "config.model_path", "evidence_id": "O01"})}}], "usage": {"total_tokens": 11}}
        with patch("infra_repair.triage_agent.http_json", return_value=response):
            result = self.run_case(agent=agent)
        self.assertTrue(result["repair_success"])
        self.assertTrue(result["root_cause_correct"])
        self.assertEqual(result["diagnosis"]["action"], "set_model_path")
        self.assertEqual(result["tool_calls"], 4)

    def test_diagnosis_rejects_correct_code_with_unrelated_or_invented_evidence(self):
        observation = {"tool": "read_logs", "result": {"logs": "File runpy line 1\nConfigured model directory does not exist"}}
        diagnosis = {"cause": "config.model_path", "object": "target", "evidence": {
            "tool": "read_logs", "field": "logs", "value": "File runpy line 1"}}
        self.assertFalse(score_diagnosis("S1", diagnosis, [observation]))
        diagnosis["evidence"]["value"] = "Configured model directory does not exist"
        self.assertTrue(score_diagnosis("S1", diagnosis, [observation]))
        self.assertFalse(score_diagnosis("S1", diagnosis, []))
        diagnosis["cause"] = "artifact.integrity"
        self.assertFalse(score_diagnosis("S1", diagnosis, [observation]))

    def test_stream_measures_tokens_and_rejects_incomplete_output(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"token_ids": [42]}\n')
                self.wfile.flush()
                if self.path == "/complete":
                    time.sleep(.05)
                    self.wfile.write(b'{"done": true}\n')
                    self.wfile.flush()
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.1.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = "http://127.0.1.1:%d" % server.server_port
            result = stream_tokens(base + "/complete", {})
            self.assertEqual(result["token_ids"], [42])
            self.assertGreater(result["client_seconds"], result["ttft_seconds"] + .03)
            with self.assertRaises(RuntimeError):
                stream_tokens(base + "/incomplete", {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_loopback_policy_does_not_allow_external_or_embedded_credentials(self):
        self.assertTrue(is_loopback_url("http://127.0.1.1:8001/v1"))
        for url in ("http://127.0.0.1.example.org", "file:///etc/passwd", "http://user:secret@localhost"):
            self.assertFalse(is_loopback_url(url))

    def test_interactive_approval_binds_action_and_rechecks_deadline(self):
        backend = FakeBackend()
        gateway = Gateway(backend, self.root / "approval.jsonl", approval="interactive")
        with patch.object(backend, "repair") as repair, patch("builtins.input", return_value="approve restore_artifacts"):
            self.assertIn("error", gateway.call("set_model_path", {"resource": "target"}))
            repair.assert_not_called()

        with patch.object(backend, "repair", return_value={"started": True}) as repair, patch("builtins.input", return_value="approve set_model_path"):
            self.assertTrue(gateway.call("set_model_path", {"resource": "target"})["started"])
            repair.assert_called_once_with("set_model_path")
        gateway.deadline = time.monotonic() + 1
        def expire(_):
            gateway.deadline = time.monotonic() - 1
            return "approve set_model_path"
        with patch.object(backend, "repair") as repair, patch("builtins.input", side_effect=expire):
            self.assertIn("error", gateway.call("set_model_path", {"resource": "target"}))
            repair.assert_not_called()

    def test_acceptance_rejects_incomplete_mixed_and_unverified_cohorts(self):
        import copy
        reference, rollback = [], []
        for scenario in ("S1", "S2", "S3", "S4", "S5"):
            for index in range(3):
                row = {"scenario": scenario, "run_id": scenario + str(index),
                       "agent": "triage-test-model", "agent_revision": "test-v1", "agent_is_reference": True,
                       "backend": "kubeedge", "contract": {"dtype": "float16"},
                       "target_artifact_identity": {"weights": "test-only"},
                       "implementation_identity": {"code": "test-only"}, "runtime_identity": {"node": "test-only"},
                       "repair_success": True, "repair_seconds": 1, "root_cause_correct": True,
                       "injection_success": True, "cleanup_success": True, "rollback_success": None,
                       "unauthorized_executions": 0, "diagnosis_scoring": "observed-evidence-v3",
                       "verification": {"measurement_scope": "streaming_token_regression"}}
                reference.append(row)
                rollback.append({**row, "run_id": "rollback-" + row["run_id"], "agent": "forced-failure",
                                 "repair_success": False, "repair_seconds": None, "rollback_success": True})
        self.assertTrue(assess(reference, rollback)["passed"])
        self.assertFalse(assess(reference[:-1], rollback)["passed"])
        mixed = copy.deepcopy(reference)
        mixed[0]["agent_revision"] = "different"
        self.assertFalse(assess(mixed, rollback)["passed"])
        rollback[0]["rollback_success"] = None
        self.assertFalse(assess(reference, rollback)["passed"])

    def test_edge_observations_probe_network_instead_of_injection_state(self):
        from types import SimpleNamespace
        backend = EdgeBackend(self.root / "run", self.root / "source", kubeconfig=self.root / "config")
        backend.cloud_ip = "172.18.0.3"
        backend.node_architecture = "amd64"
        backend.image_architectures = {backend.good_image: "amd64"}
        config = SimpleNamespace(stdout=json.dumps({"data": {"service.json": json.dumps({"revision": "R2"})}}))
        with patch.object(backend, "_command", return_value=SimpleNamespace(returncode=0)) as command, \
                patch.object(backend, "_kubectl", return_value=config), \
                patch.object(backend, "_health", return_value={"revision": "R2"}), \
                patch.object(backend, "_blocked", side_effect=AssertionError("Hidden injection state accessed")):
            state = backend.inspect()
            self.assertTrue(state["control_link_reachable"])
            self.assertNotIn("control_link_blocked", state)
            self.assertIn("/dev/tcp/$1/10000", command.call_args.args[0][-3])
    def test_local_http_ignores_proxy_and_refuses_redirect(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "https://example.invalid/private")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"local": true}')
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.1.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"http_proxy": "http://127.0.1.1:1", "no_proxy": ""}):
                base = "http://127.0.1.1:%d" % server.server_port
                self.assertEqual(http_json(base), {"local": True})
                with self.assertRaises(HTTPError) as error:
                    http_json(base + "/redirect")
                self.assertEqual(error.exception.code, 302)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
