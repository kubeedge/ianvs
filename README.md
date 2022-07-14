# Ianvs
[![CI](https://github.com/kubeedge-sedna/ianvs/workflows/CI/badge.svg?branch=v0.1)](https://github.com/kubeedge-sedna/ianvs/actions)
[![LICENSE SCAN](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fkubeedge-sedna%2Fianvs.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fkubeedge-sedna%2Fianvs?ref=badge_shield)
[![LICENSE](https://img.shields.io/github/license/kubeedge-sedna/ianvs.svg)](/LICENSE)

Ianvs is a distributed synergy AI benchmarking project incubated in KubeEdge SIG AI. Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, in order to facilitate more efficient and effective development. More detailedly, Ianvs prepares not only test cases with datasets and corresponding algorithms, but also benchmarking tools including simulation and hyper-parameter searching. Ianvs also revealing best practices for developers and end users with presentation tools including leaderboards and test reports. 

## Scope
The distributed synergy AI benchmarking Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, in order to facilitate more efficient and effective development. 

The scope of Ianvs includes
- Providing end-to-end benchmark toolkits across devices, edge nodes and cloud nodes based on typical distributed-synergy AI paradigms and applications. 
    - Tools to manage test environment. For example, it would be necessary to support the CRUD (Create, Read, Update and Delete) actions in test environments. Elements of such test environments include algorithm-wise and system-wise configuration.  
    - Tools to control test cases. Typical examples include paradigm templates, simulation tools, and hyper-parameter-based assistant tools.
    - Tools to manage benchmark presentation, e.g., leaderboard and test report generation. 
- Cooperation with other organizations or communities, e.g., in KubeEdge SIG AI, to establish comprehensive benchmarks and developed related applications, which can include but are not limited to 
    - Dataset collection, re-organization, and publication
    - Formalized specifications, e.g., standards 
    - Holding competitions or coding events, e.g., open source promotion plan
    - Maintaining solution leaderboards or certifications for commercial usage 


## Architecture
The architectures and related concepts are shown in the below figure. The ianvs is designed to run within a single node. Critical components include
- Test Environment Manager: the CRUD of test environments serving for global usage
- Test Case Controller: control the runtime behavior of test cases like instance generation and vanish 
    - Generation Assistant: assist users to generate test cases based on certain rules or constraints, e.g., the range of parameters 
    - Simulation Controller: control the simulation process of edge-cloud synergy AI, including the instance generation and vanishment of simulation containers
- Story Manager: the output management and presentation of the test case, e.g., leaderboards


![](docs/guides/images/ianvs_arch.png)

More details on Ianvs components: 
1. Test-Environment Manager supports the CRUD of Test environments, which basically includes
    - Algorithm-wise configuration
        - Public datasets
        - Pre-processing algorithms
        - Feature engineering algorithms
        - Post-processing algorithms like metric computation
    - System-wise configuration
        - Overall architecture
        - System constraints or budgets
            - End-to-end cross-node 
            - Per node
1. Test-case Controller, which includes but is not limited to the following components 
    - Templates of common distributed-synergy-AI paradigms, which can help the developer to prepare their test case without too much effort. Such paradigms include edge-cloud synergy joint inference, incremental learning, federated learning, and lifelong learning. 
    - Simulation tools. Develop simulated test environments for test cases
    - Other tools to assist test-case generation. For instance, prepare test cases based on a given range of hyper-parameters. 
1. Story Manager, which includes but is not limited to the following components
    - Leaderboard generation
    - Test report generation


## Guides

### Documents

Documentation is located on [readthedoc.io](https://ianvs.readthedocs.io/). These documents can help you understand Ianvs better.


### Installation
Follow the [Ianvs installation document](docs/guides/how-to-install-ianvs.md) to install Ianvs.

### Examples
Scenario PCB-AoI：[Industrial Defect Detection on the PCB-AoI Dataset](/examples/pcb-aoi/README.md).  
Example PCB-AoI-1：[Testing single task learning in industrial defect detection](/docs/proposals/test-reports/testing-single-task-learning-in-industrial-defect-detection-with-pcb-aoi.md).  
Example PCB-AoI-2：[Testing incremental learning in industrial defect detection](/docs/proposals/test-reports/testing-incremental-learning-in-industrial-defect-detection-with-pcb-aoi.md).  


## Roadmap

* [2022 H2 Roadmap](docs/roadmap.md)

## Meeting

Regular Community Meeting for KubeEdge SIG AI:
- Europe Time: **Thursdays at 16:30-17:30 Beijing Time** (biweekly, starting from Feb. 2022).
([Convert to your timezone.](https://www.thetimezoneconverter.com/?t=16%3A30&tz=GMT%2B8&))
- Pacific Time: **Thursdays at 10:00-11:00 Beijing Time** (biweekly, starting from Feb. 2022).
([Convert to your timezone.](https://www.thetimezoneconverter.com/?t=10%3A00&tz=GMT%2B8&))

Resources:
- [Meeting notes and agenda](https://docs.google.com/document/d/12n3kGUWTkAH4q2Wv5iCVGPTA_KRWav_eakbFrF9iAww/edit)
- [Meeting recordings](https://www.youtube.com/playlist?list=PLQtlO1kVWGXkRGkjSrLGEPJODoPb8s5FM)
- [Meeting link](https://zoom.us/j/4167237304)
- [Meeting Calendar](https://calendar.google.com/calendar/u/0/r?cid=Y19nODluOXAwOG05MzFiYWM3NmZsajgwZzEwOEBncm91cC5jYWxlbmRhci5nb29nbGUuY29t) | [Subscribe](https://calendar.google.com/calendar/u/0/r?cid=OHJqazhvNTE2dmZ0ZTIxcWlidmxhZTNsajRAZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ)

## Contact

<!--
If you need support, start with the [troubleshooting guide](./docs/troubleshooting.md), and work your way through the process that we've outlined.
-->

If you have questions, feel free to reach out to us in the following ways:
- [slack channel](https://app.slack.com/client/TDZ5TGXQW/C01EG84REVB/details)

## Contributing

If you're interested in being a contributor and want to get involved in developing the Ianvs code, please see [CONTRIBUTING](CONTRIBUTING.md) for details on submitting patches and the contribution workflow.

## License

Ianvs is under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.