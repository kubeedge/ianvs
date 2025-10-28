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
    kubeedge_version: "1.8.0"
    sedna_version: "0.4.3"
```

Related parameters and explanations are as follows:

- `cloud_number` : int, number of the cloud worker
- `edge_number` : int, number of the edge nodes.
- `cluster_name` : int, name of the simulation cluster.
- `kubeedge_version` : string, version of kubeedge, e.g. 1.8.0, latest.
- `sedna_version` : string, version of sedna, e.g. 0.4.3, latest.

Note that the current simulation environment build script is still being debugged at this time. Our current testing is based on Kubeedge v1.8.0, sedna v0.4.3, and the system OS is ubuntu 20.04.

### 2. Run the benchmarkingJob

We just need to attach the `benchmarkingJob.yaml` when executing the `ianvs` command as before. Just like `ianvs -f /somepath/benchmarkingJob.yaml`

Next, the `Simulation System Administrator` module will first check your system environment, including the following checks:

1. Whether `docker` has been installed. If not, it will try to help users install it.
2. Whether `kind` has been installed. If not, it will try to help users install it.
3. Whether the number of cpus is sufficient. Currently we tentatively need 4 CPU logical cores.
4. Whether the available memory is sufficient. More than 4GB of free memory is required.

If you pass the above environment tests, you will see the following information in the terminal.

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
