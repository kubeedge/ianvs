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


.. toctree::
    :maxdepth: 1
    :caption: INTRODUCTION
    Introduction to Ianvs <distributed-synergy-ai-benchmarking>
    guides/quick-start
    roadmap

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




RELATED LINKs
=============

.. mdinclude:: related-link.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
