# How to contribute leaderboards or test reports


This document helps you prepare environment for developing code for Ianvs.
If you follow this guide and find some problem, please fill an issue to update this file.

## 1. Install Tools
### Install Git

Ianvs is managed with [git], and to develop locally you
will need to install `git`.

You can check if `git` is already on your system and properly installed with
the following command:

```
git --version
```



## 2. Clone the code

Clone the `Ianvs` repo:

```shell
git clone http://github.com/kubeedge/ianvs.git
```


## 3. Setup Ianvs, run it

Please follow [Ianvs setup] to install Ianvs, and then run your own algorithm to output test reports.


## 4. Declare your grades 
You can compare your reports and those results on the [leaderboard]. 
if you found your result is better than those, you can put yours to the right location. 
Also, test reports is welcome to upload to [there](../proposals/test-reports).   



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
[leaderboard]: ../proposals/leaderboards/
