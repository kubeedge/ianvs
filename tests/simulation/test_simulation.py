"""Tests for the simulation config and system admin."""
import unittest
from unittest import mock

from core.testcasecontroller.simulation import Simulation
from core.testcasecontroller.simulation_system_admin import simulation_system_admin as admin


class SimulationConfigTest(unittest.TestCase):
    def test_valid(self):
        sim = Simulation({"cloud_number": 1, "edge_number": 2, "cluster_name": "c"})
        self.assertEqual((sim.cloud_number, sim.edge_number, sim.cluster_name), (1, 2, "c"))

    def test_defaults(self):
        self.assertEqual(Simulation({}).kubeedge_version, "v1.14.0")

    def test_invalid(self):
        for cfg in ({"cloud_number": True}, {"edge_number": -1}, {"cluster_name": 3}, {"cluster_name": " "},
                    {"edge_number": 10}, {"cloud_number": 3}, "x"):
            with self.assertRaises(ValueError):
                Simulation(cfg)


class AdminTest(unittest.TestCase):
    def test_missing_docker(self):
        with mock.patch.object(admin.shutil, "which", return_value=None):
            with self.assertRaises(RuntimeError):
                admin.check_host_docker()

    def test_env_versions(self):
        env = admin._installer_env(Simulation({"cluster_name": "c"}))
        self.assertEqual(env["KUBEEDGE_VERSION"], "v1.14.0")
        self.assertNotIn("SEDNA_VERSION", env)
        self.assertEqual(env["CLUSTER_NAME"], "c")

    def test_failed_build_cleans_up(self):
        sim = Simulation({"cluster_name": "c"})
        with mock.patch.object(admin, "check_host_enviroment"),                 mock.patch.object(admin, "_fetch_installer", return_value=b""),                 mock.patch.object(admin, "_run", return_value=mock.Mock(returncode=1)),                 mock.patch.object(admin, "destory_simulation_enviroment") as destroy:
            with self.assertRaises(RuntimeError):
                admin.build_simulation_enviroment(sim)
        destroy.assert_called_once_with(sim)


if __name__ == "__main__":
    unittest.main()
