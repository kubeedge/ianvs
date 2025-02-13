# How to contribute documentation

**First off, thanks for taking the time to think about contributing!**

The KubeEdge community abides by the CNCF [code of conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md). Here is an excerpt:

_As contributors and maintainers of this project, and in the interest of fostering an open and welcoming community, we pledge to respect all people who contribute through reporting issues, posting feature requests, updating documentation, submitting pull requests or patches, and other activities._

---
For Contributing to the ianvs documentation hosted at `https://ianvs.readthedocs.io/en/latest`, follow the following instructions:- 

### Forking the project

You can [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this project repository to create and submit changes to the documentation.

Then, open the terminal of your choice and run the following command to clone the forked project to your computer. The project is saved to the location where the terminal is opened from. If you want to change the location, use `cd` command to change the directory.

```shell
git clone <your forked repository link>
```
Wait until all of the project files are downloaded. A folder with the name `ianvs` will be created. You need to enter into the directory with the `cd` command.

**Note: In order to build ianvs documentation on your local machine you need a python version 3.6**.

- Now, execute these commands in your terminal 
```
cd docs
python3 conf.py # downloads the requirements
```
Now, build and serve it live using sphinx-autobuild,

```
make livehtml
```

Now, you can open http://127.0.0.1:8000 on any browser to see the rendered HTML with live updates. Clear cookies and site data in your browser window to view up to date site.