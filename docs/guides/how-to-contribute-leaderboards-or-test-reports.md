# How to contribute test reports or leaderboards

This document helps you to contribute stories, i.e., test reports or leaderboards, for Ianvs.
If you follow this guide and find some problem, it is appreciated to submit an issue to update this file.

## Test Reports

Everyone is welcome to submit and share your own test report to the community.

### 1. Setup and Testing

Ianvs is managed with [git], and to develop locally you
will need to install `git`.

You can check if `git` is already on your system and properly installed with
the following command:

```
git --version
```

Clone the `Ianvs` repo.:

```shell
git clone http://github.com/kubeedge/ianvs.git
```

Please follow the [Ianvs setup] to install Ianvs, and then run your own algorithm to output test reports.

### 2. Declare your grades

You may want to compare your testing result and those results on the [leaderboard].

Test reports are welcome after benchmarking. It can be submitted [here](https://github.com/kubeedge/ianvs/tree/main/docs/proposals/test-reports) for further review.

## Leaderboards

Leaderboards, i.e., rankings of the test object, are public for everyone to visit. Example:[leaderboard].

Except for [Ianvs Owners](https://github.com/kubeedge/ianvs/blob/main/OWNERS), there are mainly two roles for a leaderboard publication:

1. Developer: submit the test object for benchmarking, including but not limited to materials like algorithm, test case following Ianvs settings, and interfaces.
2. Maintainer: testing materials provided by developers and releasing the updated leaderboard to the public.

For potential developers,

- Develop your algorithm with ianvs and choose the algorithm to submit.
- Make sure the submitted test object runs properly under the latest version of Ianvs before submission. Maintainers are not responsible to debug for the submitted objects.
- Do NOT need to submit the new leaderboard. Maintainers are responsible to make the test environment consistent for all test objects under the same leaderboard and execute the test object to generate a new leaderboard.
- If the test object is ready, you are welcome to contact [Ianvs Owners](https://github.com/kubeedge/ianvs/blob/main/OWNERS). Ianvs owners will connect you and maintainers, in order to receive your test object. Note that when developers submit the test object, developers give maintainers the right to test them.

For potential maintainers,

- To maintain the consistency of test environments and test objects, the [leaderboard] submission is at present calling for acknowledged organizations to apply in charge. Please contact
- Maintainers should be responsible for the result submitted.
- Maintainers should update the leaderboard in a monthly manner.
- Maintainers are NOT allowed to use the test object in purpose out of Ianvs benchmarking without formal authorization from developers.
- Besides submitted objects, maintainers are suggested to test objects released in KubeEdge SIG AI or other classic solutions released in public.

[git]: https://git-scm.com/
[framework]: /docs/proposals/architecture.md#architecture
[github]: https://github.com/
[golang]: https://golang.org/doc/install
[k8s-setup]: https://kubernetes.io/docs/setup/
[k8s-tools]: https://kubernetes.io/docs/tasks/tools
[minikube]: https://minikube.sigs.k8s.io/docs/start/
[kind]: https://kind.sigs.k8s.io
[kubeedge]: https://kubeedge.io/en/docs/
[kubeedge-k8s-compatibility]: https://github.com/kubeedge/kubeedge#kubernetes-compatibility
[Ianvs Setup]: how-to-install-ianvs.md
[leaderboard]: ../leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-single-task-learning.md
