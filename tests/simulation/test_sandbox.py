"""Tests for the simulation sandbox: profile, job administrator, controller and runner."""
import importlib.util
import json
import os
import tempfile
import unittest
import uuid
from unittest import mock

from core.simulationcontroller import runner
from core.simulationcontroller import simulation_job_administrator as sja
from core.simulationcontroller.sandbox_manager import SandboxProfile
from core.simulationcontroller.simulation_controller import (
    SimulationController, mount_paths, referenced_paths)
from core.simulationcontroller.simulation_job_administrator import (
    JobFailure, SimulationJobAdministrator)


class SandboxProfileTest(unittest.TestCase):
    def test_defaults(self):
        profile = SandboxProfile({})
        self.assertTrue(profile.enabled)
        self.assertEqual((profile.image, profile.timeout), ("", 3600))
        self.assertEqual(profile.resources("singletasklearning"), (1024, 1))
        self.assertEqual(profile.resources("jointinference"), (4096, 2))

    def test_device_simulation_overrides_field_by_field(self):
        profile = SandboxProfile({}, {"max_cpu_cores": 3})
        self.assertEqual(profile.resources("lifelonglearning"), (3072, 3))
        profile = SandboxProfile({}, {"max_memory_mb": 512, "network_latency_ms": 20})
        self.assertEqual(profile.resources("federatedlearning"), (512, 4))

    def test_invalid(self):
        for config, device in (("x", None), ({"enabled": "yes"}, None),
                               ({"timeout": 0}, None), ({"requirements": "r.txt"}, None),
                               ({"data_paths": [""]}, None), ({"base_image": ""}, None),
                               ({}, {"max_memory_mb": -1}), ({}, {"max_cpu_cores": True}),
                               ({}, "x")):
            with self.assertRaises(ValueError):
                SandboxProfile(config, device)


class PathsTest(unittest.TestCase):
    def test_referenced_and_mounted_paths(self):
        with tempfile.TemporaryDirectory() as root:
            for rel in ("examples/a/model.py", "examples/a/sub/data.jsonl", "top.yaml"):
                os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
                open(os.path.join(root, rel), "w", encoding="utf-8").close()
            cwd = os.getcwd()
            os.chdir(root)
            try:
                config = {"a": {"url": "./examples/a/model.py", "n": 3, "missing": "./nope"},
                          "list": ["./examples/a/sub/data.jsonl", "top.yaml", " "]}
                paths = referenced_paths(config)
                self.assertEqual(len(paths), 3)
                mounts = mount_paths(paths, root)
            finally:
                os.chdir(cwd)
            # the directory of a file, the file itself next to the working directory,
            # and nested directories folded into the outermost one
            self.assertEqual(mounts, sorted([os.path.join(root, "examples", "a"),
                                             os.path.join(root, "top.yaml")]))


def _admin_with_nodes(nodes):
    admin = SimulationJobAdministrator("c")
    items = [{"metadata": {"name": name,
                           "labels": {sja.EDGE_ROLE_LABEL: ""} if edge else {}},
              "status": {"conditions": [{"type": "Ready",
                                         "status": "True" if ready else "False"}]}}
             for name, edge, ready in nodes]
    admin.kubectl = mock.Mock(return_value=json.dumps({"items": items}))
    return admin


class JobAdministratorTest(unittest.TestCase):
    def test_select_node(self):
        self.assertEqual(_admin_with_nodes([("c-control-plane", False, True),
                                            ("c-worker", False, True),
                                            ("edge1", True, True),
                                            ("edge0", True, False)]).select_node(), "edge1")
        self.assertEqual(_admin_with_nodes([("c-control-plane", False, True),
                                            ("c-worker", False, True)]).select_node(),
                         "c-worker")
        self.assertEqual(_admin_with_nodes([("c-control-plane", False, True)]).select_node(),
                         "c-control-plane")
        with self.assertRaises(RuntimeError):
            _admin_with_nodes([("c-worker", False, False)]).select_node()

    def test_select_node_waits_for_edge(self):
        not_ready = _admin_with_nodes([("c-worker", False, True), ("edge0", True, False)])
        ready = _admin_with_nodes([("c-worker", False, True), ("edge0", True, True)])
        admin = SimulationJobAdministrator("c")
        admin.kubectl = mock.Mock(side_effect=[not_ready.kubectl(), ready.kubectl()])
        with mock.patch.object(sja, "POLL_INTERVAL_S", 0):
            self.assertEqual(admin.select_node(), "edge0")
            # the edge node stays NotReady: a cloud node after the timeout
            with mock.patch.object(sja, "EDGE_READY_TIMEOUT_S", 0):
                self.assertEqual(not_ready.select_node(), "c-worker")

    def test_node_container(self):
        admin = SimulationJobAdministrator("c")
        self.assertEqual(admin.node_container("c-worker"), "c-worker")

        def run(cmd, **_):
            if cmd[1] == "ps":
                return mock.Mock(stdout="other-edge0 other\nc-edge0 c\nsedna-mini-edge1 \n")
            return mock.Mock(stdout={"c-edge0": "edge0", "other-edge0": "edge0"}[cmd[-1]])
        with mock.patch.object(sja, "_run", side_effect=run):
            self.assertEqual(admin.node_container("edge0"), "c-edge0")
            # an edge container of an older installer, named sedna-mini-<node>
            self.assertEqual(admin.node_container("edge1"), "sedna-mini-edge1")
            with self.assertRaises(RuntimeError):
                admin.node_container("edge2")

    def test_job_manifest(self):
        manifest = SimulationJobAdministrator.job_manifest(
            "ianvs-x", "edge0", "img:1", ["python", "-m", "m"],
            [(sja.INPUT_STAGE_ROOT + "/data", "/data"),
             (f"{sja.NODE_STAGE_ROOT}/ianvs-x/ws/out", "/ws/out")],
            (1024, 2), "/work")
        spec = manifest["spec"]["template"]["spec"]
        container = spec["containers"][0]
        self.assertEqual(manifest["spec"]["backoffLimit"], 0)
        self.assertEqual(spec["nodeSelector"], {"kubernetes.io/hostname": "edge0"})
        self.assertEqual(container["imagePullPolicy"], "Never")
        self.assertEqual(container["resources"]["limits"], {"memory": "1024Mi", "cpu": "2"})
        self.assertEqual(container["workingDir"], "/work")
        self.assertEqual([m["mountPath"] for m in container["volumeMounts"]],
                         ["/data", "/ws/out"])
        self.assertEqual([v["hostPath"]["path"] for v in spec["volumes"]],
                         [f"{sja.INPUT_STAGE_ROOT}/data", f"{sja.NODE_STAGE_ROOT}/ianvs-x/ws/out"])

    def test_build_image_with_wheel(self):
        dockerfiles = []

        def run(cmd, **_):
            # docker build <context>: read the Dockerfile before the context is removed
            with open(os.path.join(cmd[-1], "Dockerfile"), encoding="utf-8") as file:
                dockerfiles.append(file.read())
            self.assertTrue(os.path.isfile(
                os.path.join(cmd[-1], "wheels-1", "algo-0.1-py3-none-any.whl")))
        with tempfile.TemporaryDirectory() as tmp:
            requirements = os.path.join(tmp, "requirements.txt")
            wheel = os.path.join(tmp, "algo-0.1-py3-none-any.whl")
            for path in (requirements, wheel):
                open(path, "w", encoding="utf-8").close()
            profile = SandboxProfile({"enabled": True, "requirements": [requirements, wheel]})
            with mock.patch.object(sja, "_run", side_effect=run), \
                    mock.patch.object(sja.subprocess, "run", return_value=mock.Mock(returncode=1)):
                image = SimulationJobAdministrator.build_image(profile)
        self.assertTrue(image.startswith(sja.IMAGE_REPOSITORY + ":"))
        self.assertIn("pip install --no-cache-dir -r requirements-extra-0.txt", dockerfiles[0])
        self.assertIn("pip install --no-cache-dir --no-deps wheels-1/algo-0.1-py3-none-any.whl",
                      dockerfiles[0])

    def test_cri_log_text(self):
        raw = ("2026-11-12T10:00:00.1Z stdout F epoch 1\n"
               "2026-11-12T10:00:00.2Z stderr P Traceback (most \n"
               "2026-11-12T10:00:00.3Z stderr F recent call last):\n"
               "2026-11-12T10:00:00.4Z stdout F \n")
        self.assertEqual(sja.cri_log_text(raw),
                         "epoch 1\nTraceback (most recent call last):\n\n")

    def test_save_logs(self):
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "sub", "sandbox.log")
            with mock.patch.object(sja, "_run", return_value=mock.Mock(
                    stdout="2026-11-12T10:00:00Z stdout F hello\n")) as run:
                SimulationJobAdministrator.save_logs("c-edge0", "ianvs-x", path)
            self.assertIn("/var/log/pods/default_ianvs-x-*_*/testcase/*.log",
                          run.call_args[0][0][-1])
            with open(path, encoding="utf-8") as file:
                self.assertEqual(file.read(), "hello\n")
            # a missing log is not an error
            with mock.patch.object(sja, "_run", side_effect=RuntimeError("no such file")):
                SimulationJobAdministrator.save_logs("c-edge0", "ianvs-y",
                                                     os.path.join(out, "y.log"))
            self.assertFalse(os.path.exists(os.path.join(out, "y.log")))

    def test_run_job_reports_terminated_state(self):
        admin = SimulationJobAdministrator("c")
        terminated = {"reason": "Completed", "exitCode": 0, "message": "{}"}
        pods = {"items": [{"status": {"containerStatuses": [
            {"state": {"terminated": terminated}}]}}]}
        admin.kubectl = mock.Mock(side_effect=["applied", json.dumps(pods)])
        self.assertEqual(admin.run_job({"metadata": {"name": "j"}}, 10), terminated)

    def test_run_job_unpullable_image(self):
        admin = SimulationJobAdministrator("c")
        pods = {"items": [{"status": {"containerStatuses": [
            {"state": {"waiting": {"reason": "ErrImageNeverPull"}}}]}}]}
        admin.kubectl = mock.Mock(side_effect=["applied", json.dumps(pods)])
        with self.assertRaises(JobFailure) as ctx:
            admin.run_job({"metadata": {"name": "j"}}, 10)
        self.assertEqual(ctx.exception.status, "EXEC_FAILURE")

    def test_run_job_timeout(self):
        admin = SimulationJobAdministrator("c")
        admin.kubectl = mock.Mock(side_effect=["applied"] + [json.dumps({"items": []})] * 5)
        with mock.patch.object(sja, "POLL_INTERVAL_S", 0.01):
            with self.assertRaises(JobFailure) as ctx:
                admin.run_job({"metadata": {"name": "j"}}, 0.02)
        self.assertEqual(ctx.exception.status, "TIMEOUT")


class ControllerResultTest(unittest.TestCase):
    def test_success(self):
        message = json.dumps({"status": "SUCCESS", "metrics": {"accuracy": 0.9}})
        self.assertEqual(SimulationController._result(
            {"exitCode": 0, "message": message}, "/out", (1024, 1)), {"accuracy": 0.9})

    def test_result_file_fallback(self):
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, runner.RESULT_FILE), "w", encoding="utf-8") as file:
                json.dump({"status": "SUCCESS", "metrics": {"a": 1}}, file)
            message = json.dumps({"status": "SUCCESS", "result_file": runner.RESULT_FILE})
            self.assertEqual(SimulationController._result(
                {"exitCode": 0, "message": message}, out, (1024, 1)), {"a": 1})

    def test_failures(self):
        cases = (({"reason": "OOMKilled", "exitCode": 137}, "OOM_FAILURE"),
                 # as edgecore reports it
                 ({"reason": "Error", "exitCode": 137, "message": "a log line\n"},
                  "OOM_FAILURE"),
                 ({"reason": "Error", "exitCode": 1, "message": json.dumps(
                     {"status": "EXEC_FAILURE", "error": "boom"})}, "EXEC_FAILURE"),
                 ({"reason": "Error", "exitCode": 1, "message": "Traceback ..."}, "EXEC_FAILURE"))
        for terminated, status in cases:
            with self.assertRaises(JobFailure) as ctx:
                SimulationController._result(terminated, "/out", (1024, 1))
            self.assertEqual(ctx.exception.status, status)


class ControllerRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        root = self.tmp.name
        self.data = os.path.join(root, "data")
        os.makedirs(self.data)
        open(os.path.join(self.data, "train.jsonl"), "w", encoding="utf-8").close()
        os.makedirs(os.path.join(root, "configs"))
        self.testenv = os.path.join(root, "configs", "testenv.yaml")
        with open(self.testenv, "w", encoding="utf-8") as file:
            json.dump({"train_url": os.path.join(self.data, "train.jsonl")}, file)
        self.algorithm_url = os.path.join(root, "configs", "algorithm.yaml")
        with open(self.algorithm_url, "w", encoding="utf-8") as file:
            json.dump({"paradigm_type": "singletasklearning"}, file)
        self.workspace = os.path.join(root, "workspace")
        self.controller = SimulationController(
            mock.Mock(cluster_name="c"), SandboxProfile({"enabled": True}), self.testenv)
        self.controller.admin = mock.Mock()
        self.controller.container = "c-edge0"
        self.controller.node = "edge0"

    def tearDown(self):
        self.tmp.cleanup()

    def _testcase(self):
        algorithm = mock.Mock(config_url=self.algorithm_url, paradigm_type="singletasklearning",
                              modules_index=0)
        algorithm.name = "algo"
        return mock.Mock(id=uuid.uuid1(), algorithm=algorithm)

    def test_inputs_are_staged_once(self):
        admin = self.controller.admin
        admin.run_job.return_value = {"exitCode": 0, "message": json.dumps(
            {"status": "SUCCESS", "metrics": {"accuracy": 1.0}})}
        for _ in range(2):
            self.assertEqual(self.controller.run_sandboxed(self._testcase(), self.workspace),
                             {"accuracy": 1.0})
        input_calls = [c for c in admin.stage_paths.call_args_list
                       if c[0][1] == sja.INPUT_STAGE_ROOT]
        self.assertEqual(len(input_calls), 1)
        self.assertIn(self.data, input_calls[0][0][2])
        # the output directory of each test case is staged per job
        self.assertEqual(admin.stage_paths.call_count, 3)
        manifest_mounts = admin.job_manifest.call_args[0][4]
        self.assertIn((sja.INPUT_STAGE_ROOT + self.data, self.data), manifest_mounts)
        self.assertEqual(admin.save_logs.call_count, 2)

    def test_failed_job_keeps_outputs_log_and_status(self):
        admin = self.controller.admin
        admin.run_job.side_effect = JobFailure("TIMEOUT", "too slow")
        testcase = self._testcase()
        with self.assertRaises(JobFailure):
            self.controller.run_sandboxed(testcase, self.workspace)
        admin.collect.assert_called_once()
        admin.save_logs.assert_called_once()
        admin.delete_job.assert_called_once()
        out = os.path.join(self.workspace, "algo", str(testcase.id))
        self.assertEqual(admin.save_logs.call_args[0][2], os.path.join(out, "sandbox.log"))
        with open(os.path.join(out, runner.RESULT_FILE), encoding="utf-8") as file:
            self.assertEqual(json.load(file)["status"], "TIMEOUT")


class RunnerTest(unittest.TestCase):
    def test_to_jsonable(self):
        class Scalar:  # pylint: disable=too-few-public-methods
            def item(self):
                return 0.5
        self.assertEqual(runner.to_jsonable({"a": Scalar(), "b": (1, "x")}),
                         {"a": 0.5, "b": [1, "x"]})

    def test_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as out:
            log = os.path.join(out, "termination-log")
            code = runner.main(["--testenv", "/missing.yaml", "--algorithm-name", "a",
                                "--algorithm-url", "/missing.yaml", "--modules-index", "0",
                                "--testcase-id", str(uuid.uuid1()), "--workspace", out,
                                "--output-dir", out], termination_log=log)
            self.assertEqual(code, 1)
            with open(log, encoding="utf-8") as file:
                self.assertEqual(json.load(file)["status"], "EXEC_FAILURE")
            self.assertTrue(os.path.isfile(os.path.join(out, runner.RESULT_FILE)))

    @unittest.skipUnless(importlib.util.find_spec("sedna"), "needs the sedna lib")
    def test_existing_output_dir(self):
        # the host creates and mounts the output directory before the pod starts
        example = os.path.join("tests", "simulation", "sandbox_example")
        testcase_id = str(uuid.uuid1())
        with tempfile.TemporaryDirectory() as workspace:
            out = os.path.join(workspace, "threshold_classifier", testcase_id)
            os.makedirs(out)
            log = os.path.join(workspace, "termination-log")
            code = runner.main(["--testenv", os.path.join(example, "testenv", "testenv.yaml"),
                                "--algorithm-name", "threshold_classifier",
                                "--algorithm-url",
                                os.path.join(example, "testalgorithms", "algorithm.yaml"),
                                "--modules-index", "0", "--testcase-id", testcase_id,
                                "--workspace", workspace, "--output-dir", out],
                               termination_log=log)
            with open(log, encoding="utf-8") as file:
                result = json.load(file)
            self.assertEqual((code, result["status"]), (0, "SUCCESS"), result)
            self.assertEqual(result["output_dir"], out)


class RunTestcasesTest(unittest.TestCase):
    def test_sandbox_failure_is_contained(self):
        from core.testcasecontroller.testcasecontroller import TestCaseController
        controller = TestCaseController()
        controller.test_cases = [mock.Mock(id="ok"), mock.Mock(id="oom")]
        sandbox = mock.Mock()
        sandbox.run_sandboxed.side_effect = [{"accuracy": 1.0},
                                             JobFailure("OOM_FAILURE", "limit")]
        testcases, results = controller.run_testcases("/ws", sandbox)
        self.assertEqual([t.id for t in testcases], ["ok"])
        self.assertEqual(results["ok"][0], {"accuracy": 1.0})
        for testcase in controller.test_cases:
            testcase.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
