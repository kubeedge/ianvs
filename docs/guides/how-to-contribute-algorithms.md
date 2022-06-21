# How to contributrbute an algorithm to Ianvs



Before start, it should be known that Ianvs testing algorithm development depends on [Sedna Lib]. The following is recommended contribute workflow:
1. Develop by yourself: put the algorithm implementation to ianvs [examples directory] locally, for testing.  
 Here are two examples show how to development algorithm for testing in Ianvs.
   * [incremental-learning]
   * [single-task-learning]
2. Publish to everyone: submit the algorithm implementation to [Sedna repository], for sharing, then everyone can test and use your algorithm.


Also, if new Algorithm has already beed integrated to Sedna, it can be used in Ianvs directly. 



[Sedna Lib]: https://github.com/kubeedge/sedna/tree/main/lib
[incremental-learning]: ../proposals/algorithms/incremental-learning/basicIL-fpn.md
[single-task-learning]: ../proposals/algorithms/single-task-learning/fpn.md
[examples directory]: ../../../../examples
[Sedna repository]: https://github.com/kubeedge/sedna