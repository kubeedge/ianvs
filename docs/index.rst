===========================================
Welcome to Ianvs documentation!
===========================================

Ianvs is a distributed synergy AI benchmarking project incubated in KubeEdge SIG AI. According to the landing challenge survey 2022 in KubeEdge SIG AI, 
when it comes to the landing of distributed synergy AI projects, developers suffer from the lack of support on related datasets and algorithms; 
while end users are lost in the sea of mismatched solutions. 
That limits the wide application of related techniques and hinders a prosperous ecosystem of distributed synergy AI. 

Confronted with these challenges, Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, 
in order to facilitate more efficient and effective development. More detailedly, Ianvs prepares not only test cases with datasets and corresponding algorithms, 
but also benchmarking tools including simulation and hyper-parameter searching. 
Ianvs also revealing best practices for developers and end users with presentation tools including leaderboards and test reports. 

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

Start your journey on Ianvs with the following links: 

.. toctree::
    :maxdepth: 1
    :caption: Introduction

    Introduction to Ianvs <distributed-synergy-ai-benchmarking>
    guides/quick-start

.. toctree::
    :maxdepth: 1
    :caption: GUIDEs

    guides/how-to-install-ianvs
    guides/how-to-contribute-test-environments
    guides/how-to-test-algorithms
    guides/how-to-contribute-leaderboards-or-test-reports
    guides/how-to-contribute-algorithms

.. toctree::
    :maxdepth: 1
    :caption: SCENARIOs
    
    Industrial Defect Detection: PCB-AoI <proposals/scenarios/industrial-defect-detection/pcb-aoi>

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :caption: Stories

    Leaderboard: Single Task Learning on PCB-AoI <leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-single-task-learning>
    Leaderboard: Incremental Learning on PCB-AoI <leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-incremental-learning>
    Test Report: Single Task Learning on PCB-AoI <proposals/test-reports/testing-single-task-learning-in-industrial-defect-detection-with-pcb-aoi>
    Test Report: Incremental Learning on PCB-AoI <proposals/test-reports/testing-incremental-learning-in-industrial-defect-detection-with-pcb-aoi>

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :caption: ALGORITHMs

    Single Task Learning: FPN <proposals/algorithms/single-task-learning/fpn>
    Incremental Learning: BasicIL-FPN <proposals/algorithms/incremental-learning/basicIL-fpn>


.. toctree::
    :maxdepth: 1
    :caption: API REFERENCE
    :titlesonly:
    :glob:

    api/lib/*
    Python API <autoapi/lib/ianvs/index>


.. toctree::
    :maxdepth: 1
    :caption: ROADMAP

    roadmap


RELATED LINKs
=============

.. mdinclude:: related-link.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
