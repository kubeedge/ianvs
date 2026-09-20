# How to build simulation env

This document introduces how to build a edge-cloud AI simulation environment(e.g. kubeedge sedna) with just one host.

## Introduction to `simulation controller`

The `simulation controller` is the core module of system simulation. The simulation controller has been supplemented, which build and deploy local edge-cloud simulation environment with K8s.

![](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/simulation/images/simulation_controller.jpg?raw=true)

The models in `simulation controller` are as follows:

- The `Simulation System Administrator` is used to
  1. parse the system config(simulation)
  2. check the host enviroment, e.g. check if the host has installed docker, kind, and whether memory > 4GB
  3. build the simulation enviroment
  4. create and deploy the moudles needed in simulation enviroment
  5. close and delete the simulation enviroment
- The `Simulation Job Administrator` is the core module for manage the simulation job, and provides the following funcitons:
  1. build the docker images of algorithms to be tested
  2. generate the YAML file of `simulation job`
  3. deploy and delete the `simulation job` in K8s
  4. list-watch the results of `simulation job` in K8s

## Simulation System Administrator Experiment

At present, we have completed the construction of simulation environment through `Simulation System Administrator` module.

The detailed process is as follows:

### 1. Prepare the `benchmarkingJob.yaml` file

Typically, the config file `benchmarkingJob.yaml` is as follows, which represents the configuration information required for a benchmarkingJob.

```yaml
benchmarkingjob:
  # job name of benchmarking; string type;
  name: "benchmarkingjob"
  
  # the url address of job workspace that will reserve the output of tests; string type;
  # default value: "./workspace"
  workspace: "./workspace-mmlu"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml"
  
  # the configuration of test object
  test_object:
    ...

  # the configuration of ranking leaderboard
  rank:
    ...
```

We need to supplement the config of simulation in the `benchmarkingJob.yaml`, such as the following.

```yaml
benchmarkingjob:
  ...

  simulation:
    cloud_number: 1
    edge_number: 2
    cluster_name: "ianvs-simulation"
    kubeedge_version: "v1.14.0"
    sedna_version: "latest"
```

Related parameters and explanations are as follows:

- `cloud_number` : int, number of the cloud worker (0-2).
- `edge_number` : int, number of the edge nodes (0-3).
- `cluster_name` : string, name of the simulation cluster; must not be empty if given.
- `kubeedge_version` : string, version of kubeedge; default v1.14.0. Versions v1.16 and later currently fail to deploy: the Sedna all-in-one installer uses its own bundled `keadm` (v1.9.1), which downloads `devices_v1alpha2` CRD files that KubeEdge removed from release-1.16 on, so the download fails with an HTTP 404. v1.14.0 is the newest version verified to deploy end-to-end.
- `sedna_version` : string, version of sedna, e.g. v0.4.3, latest (default: resolved by the installer).

Invalid values (booleans, negative or out-of-range node counts, empty names, non-string versions) are rejected when the config is parsed. Unknown keys are ignored with a warning. The simulation environment is torn down automatically when the job ends, even if it fails.

The simulation requires a Linux host with Docker and kind installed. Ianvs no longer auto-installs them; it stops with a clear error if they are missing.

If building the environment fails, Ianvs removes the partially built cluster before reporting the error. If one test case fails, the results of the test cases that completed are still ranked; the job stops with an error only when every test case fails.

### 2. Run the benchmarkingJob

We just need to attach the `benchmarkingJob.yaml` when executing the `ianvs` command as before. Just like `ianvs -f /somepath/benchmarkingJob.yaml`

Next, the `Simulation System Administrator` module will first check your system environment, including the following checks:

1. Whether `docker` is installed and its daemon is reachable. If not, Ianvs stops with an error that links to the Docker installation guide.
2. Whether `kind` is installed and `kind version` runs. If not, Ianvs stops with an error that links to the kind quick start.
3. Whether at least 4 CPU cores are available. Ianvs counts the cores it is allowed to use, and inside a container it also applies the container's CPU limit (cgroup), so the check reflects what the simulation can really use.
4. Whether at least 4 GB of memory is available. Ianvs reads `MemAvailable` from `/proc/meminfo` (falling back to `MemFree`), and inside a container it also applies the container's memory limit (cgroup).

If you pass the above environment tests, you will see output like the following in the terminal (this example is from the original 2022 run; your timestamps and line numbers will differ).

```shell
[2022-10-29 01:12:54,544] simulation_system_admin.py(48) [INFO] - check docker successful
[2022-10-29 01:12:54,559] simulation_system_admin.py(61) [INFO] - check Kind successful
[2022-10-29 01:12:54,617] simulation_system_admin.py(130) [INFO] - check cpu successful
[2022-10-29 01:12:54,626] simulation_system_admin.py(99) [INFO] - check memory successful
```

Next, the module starts installing all-in-one environment of sedna. If all goes well, you should get the following output:

```shell
NAME                  READY   STATUS    RESTARTS   AGE
gm-5bb9c898d6-45fnv   1/1     Running   0          33s
kb-6b7897c89-ljxbb    1/1     Running   0          34s
lc-9tkdj              1/1     Running   0          33s
lc-cc5gl              1/1     Running   0          33s
lc-qnhfp              1/1     Running   0          33s
lc-tmx62              1/1     Running   0          33s
Sedna is running:
See GM status: kubectl -n sedna get deploy
See LC status: kubectl -n sedna get ds lc
See Pod status: kubectl -n sedna get pod
[I1029 01:16:56.974] Mini Sedna is created successfully
[2022-10-29 01:17:12,880] simulation_system_admin.py(170) [INFO] - Congratulation! The simulation enviroment build successful!
```

In the end. You get an all-in-one environment of sedna.

## Troubleshooting

| Message or symptom | Cause | What to do |
|---|---|---|
| `docker is not installed; install it from https://docs.docker.com/get-docker/` | Docker is not on the host. | Install Docker, then run Ianvs again. |
| `docker is installed but the daemon is not reachable; is it running?` | The Docker daemon is stopped, or your user cannot access it. | Start the daemon (for example `sudo systemctl start docker`) and make sure your user may run `docker version`. |
| `kind is not installed; install it from https://kind.sigs.k8s.io/docs/user/quick-start/` | kind is not on the host. | Install kind, then run Ianvs again. |
| `failed to download the installer from ...` | The Sedna all-in-one installer could not be downloaded. | Check the host's network access to `raw.githubusercontent.com`; building the environment needs a network connection. |
| `cannot read host memory info (simulation requires a Linux host)` | Ianvs is not running on Linux. | Run the simulation on a Linux host. |
| `The current free memory is insufficient ...` or `The number of cpus is insufficient ...` | The host, or the container Ianvs runs in, has less than 4 GB of available memory or fewer than 4 CPU cores. | Free memory, raise the container's limits, or use a larger host. The message shows the current and required values. |
| The build fails with an HTTP 404 while `keadm` downloads a device CRD file | `kubeedge_version` is set to v1.16 or later; the installer's bundled `keadm` v1.9.1 requests CRD files those releases no longer have. | Use the default `v1.14.0`, the newest version verified to deploy end-to-end. |
