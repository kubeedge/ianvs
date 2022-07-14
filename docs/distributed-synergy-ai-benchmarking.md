# Distributed Synergy AI Benchmarking
Edge computing emerges as a promising technical framework to overcome the challenges in cloud computing. In this machine-learning era, the AI application becomes one of the most critical types of applications on the edge. Driven by the increasing computation power of edge devices and the increasing amount of data generated from the edge, edge-cloud synergy AI and distributed synergy AI techniques have received more and more attention for the sake of device, edge, and cloud intelligence enhancement. 

Nevertheless, distributed synergy AI is at its initial stage. For the time being, the comprehensive evaluation standard is not yet available for scenarios with various AI paradigms on all three layers of edge computing systems. According to the landing challenge survey 2022, developers suffer from the lack of support on related datasets and algorithms; while end users are lost in the sea of mismatched solutions. That limits the wide application of related techniques and hinders a prosperous ecosystem of distributed synergy AI. A comprehensive end-to-end distributed synergy AI benchmark suite is thus needed to measure and optimize the systems and applications. 

Ianvs thus provides a basic benchmark suite for distributed synergy AI, so that AI developers and end users can benefit from efficient development support and best practice discovery.

## Goals
For developers or end users of distributed synergy AI solutions, the goals of the distributed synergy AI framework are: 
- Facilitating efficient development for developers by preparing
    - test cases including dataset and corresponding tools
    - benchmarking tools including simulation and hyper-parameter searching
- Revealing best practices for developers and end users
    - presentation tools including leaderboards and test reports


## Scope
The distributed synergy AI benchmarking ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, in order to facilitate more efficient and effective development. 

The scope of ianvs includes
- Providing end-to-end benchmark toolkits across devices, edge nodes and cloud nodes based on typical distributed-synergy AI paradigms and applications. 
    - Tools to manage test environment. For example, it would be necessary to support the CRUD (Create, Read, Update and Delete) actions in test environments. Elements of such test environments include algorithm-wise and system-wise configuration.  
    - Tools to control test cases. Typical examples include paradigm templates, simulation tools, and hyper-parameter-based assistant tools.
    - Tools to manage benchmark presentation, e.g., leaderboard and test report generation. 
- Cooperation with other organizations or communities, e.g., in KubeEdge SIG AI, to establish comprehensive benchmarks and developed related applications, which can include but are not limited to 
    - Dataset collection, re-organization, and publication
    - Formalized specifications, e.g., standards 
    - Holding competitions or coding events, e.g., open source promotion plan
    - Maintaining solution leaderboards or certifications for commercial usage 

Targeting users
- Developers: Build and publish edge-cloud collaborative AI solutions efficiently from scratch
- End users: view and compare distributed synergy AI capabilities of solutions

The scope of ianvs does NOT include to
- Re-invent existing edge platform, i.e., kubeedge, etc.
- Re-invent existing AI framework, i.e., tensorflow, pytorch, mindspore, etc. 
- Re-invent existing distributed synergy AI framework, i.e., kubeedge-sedna, etc.
- Re-invent existing UI or GUI toolkits, i.e., prometheus, grafana, matplotlib, etc.

## Design Details
### Architecture and Modules
The architectures and related concepts are shown in the below figure. The ianvs is designed to run within a single node. Critical components include
- ``Test Environment Manager``: the CRUD of test environments serving for global usage
- ``Test Case Controller``: control the runtime behavior of test cases like instance generation and vanish 
    - ``Generation Assistant``: assist users to generate test cases based on certain rules or constraints, e.g., the range of parameters 
    - ``Simulation Controller``: control the simulation process of edge-cloud synergy AI, including the instance generation and vanishment of simulation containers
- ``Story Manager``: the output management and presentation of the test case, e.g., leaderboards

![](guides/images/ianvs_arch.png)

Ianvs includes Test-Environment Management, Test-case Controller and Story Manager in the Distributed Synergy AI benchmarking toolkits, where
1. Test-Environment Manager basically includes
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
2. Test-case Controller includes but is not limited to the following components 
    - Templates of common distributed-synergy-AI paradigms, which can help the developer to prepare their test case without too much effort. Such paradigms include edge-cloud synergy joint inference, incremental learning, federated learning, and lifelong learning. 
    - Simulation tools. Develop simulated test environments for test cases
      - Note that simulation tools are not yet available in early versions until v0.5
      - It is NOT in scope of this open-sourced Ianvs to simulate different hardware devices, e.g., simulating NPU with GPU and even CPU
    - Other tools to assist test-case generation. For instance, prepare test cases based on a given range of hyper-parameters. 
3. Story Manager includes but is not limited to the following components
    - Leaderboard generation
    - Test report generation


### Definitions of Objects

Quite a few terms exist in ianvs, which include the detailed modules and objects. To facilitate easier concept understanding, we show a hierarchical table of terms in the following figures, where the top item contains the items below it.  
![](guides/images/ianvs_concept.png)

The concept definition of modules has been shown in the Architecture Section. In the following, we introduce the concepts of objects for easier understanding. 
- ``Benchmark``: standardized evaluation process recognized by the academic or industry.  
- ``Benchmarking Job``: the serving instance for an individual benchmarking with ianvs, which takes charge of the lifetime management of all possible ianvs components. 
    - Besides components, a benchmarking job includes instances of a test environment, one or more test cases, a leaderboard, or a test report. 
    - Different test environments lead to different benchmarking jobs and leaderboards. A benchmarking job can include multiple test cases. 
- ``Test Object``: the targeted instance under benchmark testing. A typical example would be a particular algorithm or system. 
- ``Test Environment``: setups or configurations for benchmarking, typically excluding the test object.  
    - It can include algorithm-wise and system-wise configurations.  
    - It serves as the unique descriptor of a benchmarking job. Different test environments thus lead to different benchmarking jobs.
- ``Test Case``: the executable instance to evaluate the performance of the test object under a particular test environment. Thus, the test case is usually generated with a particular test environment and outputs testing results if executed. 
    - It is the atomic unit of a benchmark. That is, a benchmarking job can include quite a few test cases.
- ``Attribute (Attr.) of Test Case``: Attributes or descriptors of a test case, e.g., id, name, and time stamp.   
- ``Algorithm Paradigm``: acknowledged AI process which usually includes quite a few modules that can be implemented with replaceable algorithms, e.g., federated learning which includes modules of local train and global aggregation.  
- ``Algorithm Module``: the component of the algorithm paradigm, e.g., the global aggregation module of the federated learning paradigm.  
- ``Leaderboard``: the ranking of the test object under a specific test environment. 
    - The local node holds the local leaderboard for private usage. 
    - The global leaderboard is shared (e.g., via GitHub) by acknowledge organization. 
- ``Test Report``: the manuscript recording how the testing is conducted. 





