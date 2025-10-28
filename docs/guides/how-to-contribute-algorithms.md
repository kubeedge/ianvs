# How to contributrbute an algorithm to Ianvs

Ianvs serves as testing tools for test objects, e.g., algorithms. Ianvs does NOT include code directly on the test object. Algorithms serve as typical test objects in Ianvs and detailed algorithms are thus NOT included in this Ianvs python file. As for the details of example test objects, e.g., algorithms, please refer to third party packages in the Ianvs example. For example, for AI workflow and interface please refer to sedna and for module implementation please refer to third party packages like FPN_TensorFlow and Sedna IBT algorithm.

For algorithm contributors, you can:

1. Release a repo independent of ianvs, but the interface should still follow the SIG AI algorithm interface to launch ianvs. Here are few examples showing how to develop an algorithm for testing in Ianvs:
    * [cloud-edge-collaborative-inference-for-llm]
    * [single-task-learning]
    * [incremental-learning]
2. Integrate the targeted algorithm into sedna so that ianvs can use it directly. In this case, you can connect with sedna owners for help.

Also, if a new algorithm has already been integrated into Sedna, it can be used in Ianvs directly.

[Sedna Lib]: https://github.com/kubeedge/sedna/tree/main/lib
[incremental-learning]: ../proposals/algorithms/incremental-learning/basicIL-fpn.md
[single-task-learning]: ../proposals/algorithms/single-task-learning/fpn.md
[examples directory]: ../../../../examples
[Sedna repository]: https://github.com/kubeedge/sedna
[cloud-edge-collaborative-inference-for-llm]: https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm