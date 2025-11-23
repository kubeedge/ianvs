<<<<<<< HEAD
# How to contribute examples

## Overall contribution workflow

1. Apply for a topic.
   Once you have a new example, you can apply for a topic to discuss it on [SIG AI weekly meeting](http://github.com/kubeedge/ianvs.git).
2. Submit proposal.
   After the idea is fully discussed, the former proposal PR is needed to submit to the [Ianvs repository](http://github.com/kubeedge/ianvs.git).
3. Fix proposal review comments.  
   If other Ianvs maintainers leave review comments to the PR, you need to fix them and get at least 2 reviewers' `/lgtm`, and 1 approver's `/approve`.
4. Submit code.
   Then you can implement your code, and a good code style is encouraged.
5. Fix code review comments.  
   Besides the merge requirements of the proposal, CI passing is needed before reviewing this step.

## Add a new example

The new example should be stored in the following path:

~~~bash
examples/dataset_name/algorithm_name/task_name/
~~~

Here is an example:

~~~bash
examples/robot/lifelong_learning_bench/semantic-segmentation/
~~~

For contributing a new example, you can follow these steps to determine its storage path:

1. Under the examples directory, choose the dataset you used in this example, such as cityscapes, robot, and so on. Only when you use a new dataset can you create a new folder under the examples directory to store the example related to that dataset.
2. After determining the dataset, select the algorithm paradigm you used, such as lifelong learning, single-task learning, and so on. If you used a new algorithm paradigm, you can create a new folder under the dataset directory to store examples of that type of algorithm.
3. After determining the algorithm paradigm, select the task for your example, such as semantic segmentation, curb detection, and so on. If you used a new task, you can create a new folder under the algorithm paradigm directory to store examples of that type of task.
=======
version https://git-lfs.github.com/spec/v1
oid sha256:4380f9573f1953c143e3b3314653206c56b07fdfd55c98ccf78a7db1305ea6ae
size 1917
>>>>>>> 9676c3e (ya toh aar ya toh par)
