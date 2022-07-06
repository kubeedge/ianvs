# How to contributrbute an algorithm to Ianvs

Ianvs serves as testing tools for test objects, e.g., algorithms. Ianvs does NOT include code directly on test object. Algorithms serve as typical test objects in Ianvs and detailed algorithms are thus NOT included in this Ianvs python file. As for the details of example test objects, e.g., algorithms, please refer to third party packages in Ianvs example. For example, AI workflow and interface please refer to sedna and module implementation please refer to third party package like FPN_TensorFlow and Sedna IBT algorithm.


For algorithm contributors, you can:
1. Release a repo independent of ianvs, but interface should still follow the SIG AI algorithm interface to launch ianvs.
   Here are two examples show how to development algorithm for testing in Ianvs.
    * [incremental-learning]
    * [single-task-learning]
2. Integrated the targeted algorithm into sedna so that ianvs can use directly. in this case, you can connect sedna owners for help.


Also, if new algorithm has already bee integrated to Sedna, it can be used in Ianvs directly. 



[Sedna Lib]: https://github.com/kubeedge/sedna/tree/main/lib
[incremental-learning]: ../proposals/algorithms/incremental-learning/basicIL-fpn.md
[single-task-learning]: ../proposals/algorithms/single-task-learning/fpn.md
[examples directory]: ../../../../examples
[Sedna repository]: https://github.com/kubeedge/sedna