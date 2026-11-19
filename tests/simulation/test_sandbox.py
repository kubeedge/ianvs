"""Tests for the simulation sandbox: profile, job administrator, controller and runner."""
import importlib.util
import json
from datetime import datetime, timezone
import os
import tempfile
import unittest
import uuid
from unittest import mock

from core.simulationcontroller import runner
from core.simulationcontroller import simulation_controller as sc
from core.simulationcontroller import simulation_job_administrator as sja
from core.simulationcontroller import system_metrics_profiler as smp
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
                               ({}, "x"), ({"system_metrics": "some"}, None),
                               ({"system_metrics": ["peak_memory_mb", "nope"]}, None),
                               ({}, {"network_bandwidth_mbps": 0}),
                               ({}, {"network_latency_ms": -5})):
            with self.assertRaises(ValueError):
                SandboxProfile(config, device)

    def test_network_and_system_metrics(self):
        self.assertEqual(SandboxProfile({}).network(), (100, 0))
        profile = SandboxProfile({"system_metrics": ["peak_memory_mb", "wall_time_s"]},
                                 {"network_bandwidth_mbps": 50, "network_latency_ms": 0})
        self.assertEqual(profile.network(), (50, 0))
        self.assertEqual(profile.system_metrics, ["peak_memory_mb", "wall_time_s"])
        self.assertEqual(SandboxProfile({"system_metrics": []}).system_metrics, [])


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
            {"status": "SUCCESS", "metrics": {"accuracy": 1.0}, "wall_time_s": 12.5})}
        for _ in range(2):
            result = self.controller.run_sandboxed(self._testcase(), self.workspace)
            # the test case's metrics, its status, then the system metrics
            self.assertEqual(list(result)[:3], ["accuracy", "status", "wall_time_s"])
            self.assertEqual((result["accuracy"], result["status"], result["wall_time_s"]),
                             (1.0, "SUCCESS", 12.5))
            self.assertEqual(result["network_profile"], "100mbps-0ms")
            self.assertEqual(result["gpu_utilization_avg"], "N/A")
        # the runner waits for the network emulation
        self.assertIn("--wait-network", admin.job_manifest.call_args[0][3])
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
            record = json.load(file)
        self.assertEqual(record["status"], "TIMEOUT")
        self.assertIn("peak_memory_mb", record["system_metrics"])

    def test_oom_failure_carries_system_metrics(self):
        admin = self.controller.admin
        admin.run_job.return_value = {"exitCode": 137, "reason": "Error",
                                      "message": "a log line"}
        with self.assertRaises(JobFailure) as ctx:
            self.controller.run_sandboxed(self._testcase(), self.workspace)
        self.assertEqual(ctx.exception.status, "OOM_FAILURE")
        # killed at its limit (single-task default 1024 MiB), even without a sample
        self.assertTrue(ctx.exception.metrics["oom_failure"])
        self.assertEqual(ctx.exception.metrics["peak_memory_mb"], 1024)

    def test_selected_system_metrics(self):
        self.controller.profile = SandboxProfile({"system_metrics": ["peak_memory_mb"]})
        self.controller.admin.run_job.return_value = {"exitCode": 0, "message": json.dumps(
            {"status": "SUCCESS", "metrics": {"accuracy": 1.0}})}
        result = self.controller.run_sandboxed(self._testcase(), self.workspace)
        self.assertEqual(sorted(result), ["accuracy", "peak_memory_mb", "status"])


def _sample(t, mem, cpu_usec, io=(0, 0), net=(0, 0), du_kb=0, peak=None):
    return (f"t={t}\nmem={mem}\npeak={peak or mem}\ncpu={cpu_usec}\nio={io[0]} {io[1]}\n"
            f"net={net[0]} {net[1]}\ndu={du_kb}\n")


class ProfilerTest(unittest.TestCase):
    def test_parse_sample(self):
        self.assertEqual(smp.parse_sample("t=1.5\nmem=100\nio=10 20\nnet=\ndu=x\npeak=\n"),
                         {"t": 1.5, "mem": 100.0, "io_in": 10.0, "io_out": 20.0})

    def test_metrics(self):
        mib = smp.MB
        profiler = smp.SystemMetricsProfiler((1024, 2), (50, 20))
        # 10 s, 2 CPU cores: 5 CPU-s in the first 5 s (0.5), 9 more in the next 5 s (0.9)
        for text in (_sample(0, 100 * mib, 0, net=(0, 0), du_kb=1024),
                     _sample(5, 300 * mib, 5e6, io=(10 * mib, 0), net=(1e6, 1e6), du_kb=4096),
                     _sample(10, 200 * mib, 14e6, io=(20 * mib, 5 * mib), net=(2e6, 3e6),
                             du_kb=2048, peak=512 * mib)):
            profiler.add(smp.parse_sample(text))
        with tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, "model.pth"), "wb") as file:
                file.write(b"0" * mib)
            open(os.path.join(out, "notes.txt"), "w", encoding="utf-8").close()
            metrics = profiler.metrics({}, False, out)
        self.assertEqual(metrics["wall_time_s"], 10)
        self.assertEqual(metrics["cpu_utilization_avg"], 0.7)
        self.assertEqual(metrics["cpu_utilization_max"], 0.9)
        self.assertEqual(metrics["cpu_cores_used_avg"], 1.4)
        self.assertEqual(metrics["peak_memory_mb"], 512)
        self.assertEqual(metrics["memory_utilization_avg"], round(200 / 1024, 4))
        self.assertEqual((metrics["disk_read_bytes"], metrics["disk_write_bytes"]),
                         (20 * mib, 5 * mib))
        self.assertEqual(metrics["disk_read_mb_s_avg"], 2.0)
        self.assertEqual(metrics["disk_usage_peak_mb"], 4)
        self.assertEqual(metrics["model_artifact_size_mb"], 1)
        # 5 MB in 10 s = 4 Mbit/s; the busiest interval moves 3 MB in 5 s = 4.8 Mbit/s
        self.assertEqual(metrics["network_bandwidth_mbps_avg"], 4.0)
        self.assertEqual(metrics["network_bandwidth_mbps_peak"], 4.8)
        self.assertEqual((metrics["network_latency_ms"], metrics["network_profile"]),
                         (20, "50mbps-20ms"))
        self.assertFalse(metrics["oom_failure"])
        self.assertEqual(metrics["gpu_energy_joules"], "N/A")
        self.assertEqual(list(metrics), list(smp.SYSTEM_METRIC_NAMES))

    def test_rates_use_the_container_lifetime(self):
        # 1.8 CPU-s in a 3 s container on 1 core, of which the test case itself ran 1.4 s
        profiler = smp.SystemMetricsProfiler((1024, 1), (100, 0))
        profiler.add(smp.parse_sample(_sample(0, 1, 0)))
        profiler.add(smp.parse_sample(_sample(2, 1, 1.8e6)))
        metrics = profiler.metrics({"startedAt": "2026-11-12T08:00:00Z",
                                    "finishedAt": "2026-11-12T08:00:03Z"}, False, "", 1.4)
        self.assertEqual(metrics["wall_time_s"], 1.4)
        self.assertEqual(metrics["cpu_cores_used_avg"], 0.6)
        self.assertLessEqual(metrics["cpu_utilization_avg"], 1.0)

    def test_wall_time_sources(self):
        profiler = smp.SystemMetricsProfiler((1024, 1), (100, 0))
        self.assertEqual(profiler.wall_time({}, 3.25), 3.25)
        container = {"startedAt": "2026-11-12T08:00:00Z", "finishedAt": "2026-11-12T08:01:30Z"}
        self.assertEqual(profiler.wall_time(container), 90)
        self.assertEqual(profiler.wall_time({}), 0.0)
        # a killed test case: from when its network was ready (08:01:00) to the finish
        profiler.started = datetime(2026, 11, 12, 8, 1, tzinfo=timezone.utc).timestamp()
        self.assertEqual(profiler.wall_time(container), 30)
        self.assertEqual(profiler.container_span(container), 90)

    def test_select_metrics(self):
        metrics = {"a": 1, "b": 2}
        self.assertEqual(smp.select_metrics(metrics, "all"), metrics)
        self.assertEqual(smp.select_metrics(metrics, ["b", "c"]), {"b": 2})
        self.assertEqual(smp.select_metrics(metrics, []), {})


class ObserverTest(unittest.TestCase):
    def _observer(self):
        admin = mock.Mock()
        admin.pod_cgroup.return_value = "/sys/fs/cgroup/kubepods.slice/kubepods-podx.slice"
        admin.pod_pid.return_value = "42"
        admin.pod_ip.return_value = "10.0.0.5"
        admin.sample_pod.return_value = _sample(1, 100, 10)
        profiler = smp.SystemMetricsProfiler((1024, 1), (50, 20))
        return admin, profiler, sc._JobObserver(  # pylint: disable=protected-access
            admin, "c-edge0", "/stage/job", "/ws/out", profiler)

    def test_network_once_then_samples(self):
        admin, profiler, observer = self._observer()
        # the edge node still reports the pod as Pending, but it runs on the node
        pending = {"metadata": {"uid": "u"}, "status": {"phase": "Pending"}}
        for _ in range(3):
            observer.on_poll(pending)
        admin.apply_network.assert_called_once_with("c-edge0", "42", "10.0.0.5", 50, 20)
        admin.signal_network_ready.assert_called_once_with("c-edge0", "/stage/job", "/ws/out")
        self.assertEqual(len(profiler.samples), 3)
        # the test case starts with the first sample after the network is ready
        self.assertEqual(profiler.started, 1.0)

    def test_waits_for_the_pod_and_its_network(self):
        admin, profiler, observer = self._observer()
        pod = {"metadata": {"uid": "u"}, "status": {"phase": "Pending"}}
        admin.pod_cgroup.side_effect = [RuntimeError("not found"), admin.pod_cgroup.return_value]
        admin.pod_ip.side_effect = ["", "10.0.0.5"]
        observer.on_poll(pod)   # not on the node yet
        observer.on_poll(pod)   # on the node, no IP yet
        admin.apply_network.assert_not_called()
        self.assertEqual(profiler.samples, [])
        observer.on_poll(pod)
        admin.apply_network.assert_called_once()
        self.assertEqual(len(profiler.samples), 1)

    def test_network_failure_fails_the_job(self):
        admin, _, observer = self._observer()
        admin.apply_network.side_effect = RuntimeError("tc: not found")
        with self.assertRaises(JobFailure) as ctx:
            observer.on_poll({"metadata": {"uid": "u"},
                              "status": {"phase": "Running", "podIP": "10.0.0.5"}})
        self.assertEqual(ctx.exception.status, "EXEC_FAILURE")
        self.assertIn("network emulation", str(ctx.exception))


class NetworkAndSamplingCommandsTest(unittest.TestCase):
    def test_apply_network(self):
        with mock.patch.object(sja, "_run", return_value=mock.Mock(stdout="veth1\n")) as run:
            SimulationJobAdministrator.apply_network("c-edge0", "42", "10.0.0.5", 50, 20)
        script = run.call_args[0][0][-1]
        self.assertIn("nsenter -t 42 -n tc qdisc replace dev eth0 root netem delay 20ms "
                      "rate 50mbit", script)
        self.assertIn("ip -o route get 10.0.0.5", script)
        self.assertIn('tc qdisc replace dev "$dev" root netem rate 50mbit', script)
        with mock.patch.object(sja, "_run", return_value=mock.Mock(stdout="veth1\n")) as run:
            SimulationJobAdministrator.apply_network("c-edge0", "42", "10.0.0.5", 100, 0)
        self.assertNotIn("delay", run.call_args[0][0][-1])

    def test_pod_cgroup(self):
        found = "/sys/fs/cgroup/kubepods.slice/kubepods-poda_b.slice\n"
        with mock.patch.object(sja, "_run", return_value=mock.Mock(stdout=found)) as run:
            self.assertEqual(SimulationJobAdministrator.pod_cgroup("c", "a-b"), found.strip())
        self.assertIn("*poda_b*", run.call_args[0][0][-1])
        with mock.patch.object(sja, "_run", return_value=mock.Mock(stdout="")):
            with self.assertRaises(RuntimeError):
                SimulationJobAdministrator.pod_cgroup("c", "a-b")


class RunnerNetworkTest(unittest.TestCase):
    def test_wait_for_network(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertFalse(runner.wait_for_network(out, timeout=0.05, interval=0.01))
            open(os.path.join(out, runner.NETWORK_READY_FILE), "w", encoding="utf-8").close()
            self.assertTrue(runner.wait_for_network(out, timeout=1, interval=0.01))
            # the marker is removed, so it is not copied back with the outputs
            self.assertFalse(os.path.exists(os.path.join(out, runner.NETWORK_READY_FILE)))

    def test_network_not_ready_is_reported(self):
        with tempfile.TemporaryDirectory() as out:
            log = os.path.join(out, "termination-log")
            with mock.patch.object(runner, "wait_for_network", return_value=False):
                code = runner.main(["--testenv", "t", "--algorithm-name", "a",
                                    "--algorithm-url", "a", "--modules-index", "0",
                                    "--testcase-id", "x", "--workspace", out,
                                    "--output-dir", out, "--wait-network"],
                                   termination_log=log)
            self.assertEqual(code, 1)
            with open(log, encoding="utf-8") as file:
                self.assertIn("network emulation", json.load(file)["error"])

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
    def test_sandbox_failure_is_ranked(self):
        from core.testcasecontroller.testcasecontroller import TestCaseController
        controller = TestCaseController()
        controller.test_cases = [mock.Mock(id="ok"), mock.Mock(id="oom"),
                                 mock.Mock(id="broken")]
        sandbox = mock.Mock()
        oom = JobFailure("OOM_FAILURE", "limit")
        oom.metrics = {"peak_memory_mb": 1024, "oom_failure": True}
        sandbox.run_sandboxed.side_effect = [{"accuracy": 1.0, "status": "SUCCESS"}, oom,
                                             RuntimeError("could not stage")]
        testcases, results = controller.run_testcases("/ws", sandbox)
        # a failure with a status becomes a leaderboard row; other errors do not
        self.assertEqual([t.id for t in testcases], ["ok", "oom"])
        self.assertEqual(results["oom"][0], {"status": "OOM_FAILURE", "peak_memory_mb": 1024,
                                             "oom_failure": True})
        for testcase in controller.test_cases:
            testcase.run.assert_not_called()

    def test_host_failure_is_not_ranked(self):
        from core.testcasecontroller.testcasecontroller import TestCaseController
        controller = TestCaseController()
        ok, bad = mock.Mock(id="ok"), mock.Mock(id="bad")
        ok.run.return_value = {"accuracy": 1.0}
        bad.run.side_effect = JobFailure("EXEC_FAILURE", "x")
        controller.test_cases = [ok, bad]
        testcases, _ = controller.run_testcases("/ws")
        self.assertEqual([t.id for t in testcases], ["ok"])


class RankTest(unittest.TestCase):
    def test_failure_rows_and_status_column(self):
        from core.storymanager.rank.rank import Rank
        rank = Rank({"sort_by": [{"accuracy": "descend"}],
                     "selected_dataitem": {"paradigms": ["all"], "modules": ["all"],
                                           "hyperparameters": ["all"],
                                           "metrics": ["accuracy", "peak_memory_mb"]}})

        def testcase(name):
            case = mock.Mock(id=name, output_dir=f"/ws/{name}")
            case.algorithm.name, case.algorithm.paradigm_type = "algo", "singletasklearning"
            case.algorithm.modules = {"basemodel": mock.Mock(hyperparameters={"mode": name})}
            case.algorithm.modules["basemodel"].name = "Model"
            return case

        cases = [testcase("oom"), testcase("ok")]
        results = {"oom": ({"status": "OOM_FAILURE", "peak_memory_mb": 1024.0}, "t1"),
                   "ok": ({"accuracy": 0.9, "status": "SUCCESS", "peak_memory_mb": 310.5}, "t2")}
        with tempfile.TemporaryDirectory() as out:
            rank.save(cases, results, out)
            with open(os.path.join(out, "rank", "selected_rank.csv"), encoding="utf-8") as file:
                lines = file.read().splitlines()
        self.assertTrue(lines[0].startswith("rank,algorithm,status,accuracy,peak_memory_mb"))
        # the successful test case ranks first; the failed one follows, with N/A
        self.assertIn("SUCCESS,0.9,310.5", lines[1])
        self.assertIn("OOM_FAILURE,N/A,1024.0", lines[2])


if __name__ == "__main__":
    unittest.main()
