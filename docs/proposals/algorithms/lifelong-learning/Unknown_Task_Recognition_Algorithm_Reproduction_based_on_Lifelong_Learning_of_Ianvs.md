<<<<<<< HEAD
# Unknown Task Recognition Algorithm Reproduction based on Lifelong Learning of Ianvs



## 1 Project Background

The current mainstream machine learning paradigm is to run machine learning algorithms on a given set of data to generate a model, and then apply this model to a task in a real environment, which we can call "isolated learning". The main problem with this learning paradigm is that the model does not retain and accumulate previously learned knowledge and cannot use it in future learning, and the learning environment is static and closed, in contrast to the human learning process. In reality, the situation is so varied that it is clearly impossible to label every possible task or to collect large amounts of data before training in order for a machine learning algorithm to learn. Lifelong machine learning was created to address these problems.

Lifelong learning has five key characteristics.

1. a process of continuous learning.
2. the accumulation and retention of knowledge in the knowledge base.
3. the ability to use accumulated learned knowledge to aid future learning
4. the ability to discover new tasks.
5. the ability to learn while working.

Relying on the lifelong learning system built by KubeEdge+Sedna+Ianvs distributed collaborative AI joint inference framework, the core task of this project is to complete the unknown task identification algorithm module and embed it in the framework, with the aim of equipping the system with the ability to discover new tasks.

Traditional machine learning performs test set inference by training known samples, whose knowledge is limited, and the resulting models cannot effectively identify unknown samples in new classes, which will be treated as known samples. In a real production environment, it is difficult to guarantee that the training set contains samples from all classes. If the unknown class samples cannot be identified, the accuracy and confidence of the model will be greatly affected, and the cost consumed for model improvement is incalculable. This project aims to reproduce the algorithm of the CVPR2021 paper "Learning placeholders for open-set recognition". The paper proposes placeholders that mimic the emergence of new classes, thus helping to transform closed training into open training to accomplish recognition of unknown classes of data.

In this project, the algorithm is packaged as a python callable module and embedded in the lib library of Ianvs' lifelong learning testing system. The algorithm developer does not need to develop additional algorithms for unknown task recognition and can directly test the performance of the currently developed algorithms in combination with the dataset and testing environment provided by Ianvs. At the same time, Ianvs provides local and cloud-based algorithm performance rankings for developers to facilitate the exchange of lifelong machine learning researchers and thus promote the development of the lifelong learning research field.



## 2 Goals

1. Reproducing the algorithm from the CVPR2021 paper "Learning placeholders for open-set recognition".
2. Achieving an accuracy of 0.9 for unknown task recognition.
3. Making unknown category recognition modules according to Ianvs architecture and contextual relationship to store in the lib library, so that the algorithm can be used on demand.



## 3 Proposal

The goal of this project based on Ianvs' lifelong learning replication of the unknown task recognition algorithm is to train the classifier model by the unknown task recognition algorithm after the dataset has been trained by the initial task definition and business model in Ianvs' lifelong machine learning system, and to reason that the dataset can categorize and classify the known samples and identify the unknown samples for subsequent task assignment by reasoning with the classifier model.

This project needs to complete the task definition part and the unknown task identification part.

Task definition is the predecessor task, the business model is trained by the RFNet semantic segmentation algorithm, and the datasets involved are Cityscapes (camera data) and SYNTHIA-RAND-CITYSCAPES (simulation data).

Unknown task identification is the core task of this project. The unknown task recognition is data inference, using Wide-ResNet network structure, combined with the classification placeholders and data placeholders in the CVPR2021 paper "Learning placeholders for open-set recognition" to form a neural network and train the classifier, which can classify known samples and identify the unknown samples.

### 3.1 Architecture

This project mainly implements the function of discovering new tasks.

<img src="images/Lifelong_learning_structure.png" alt="Lifelong_learning_structure" style="zoom:48%;" />

The following is the architecture diagram of this project system, as shown in the figure, the input of the unknown sample identification module is the inference dataset, and the output bits are the known sample set and unknown sample set. Before the unknown sample identification module is the task definition module, after the task definition, the initialized dataset is considered a known dataset. In the task assignment module after the unknown sample identification module, the known data are inferred and the unknown data are re-posted and incorporated into the edge-side knowledge base and then into the cloud-side knowledge base.

<img src="images/Algorithmflow.jpeg" style="zoom:48%;" />



#### 3.1.1 How the algorithm work in ianvs

<img src="images/merge.png" alt="User" style="zoom:48%;" />

#### 3.1.2 Description of the callable class

The detailed code can be viewed [here](https://github.com/Frank-lilinjie/ianvs/blob/feature-lifelong-n/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/unseen_sample_recognition_by_scene.py)

1. Sign up in class factory from sedna

``` @ClassFactory.register(ClassType.UTD, alias="UnseenSampleRecognitionByScene")``` 

2. Get model path from [rfnet_algorithm.yaml](https://github.com/Frank-lilinjie/ianvs/blob/feature-lifelong-n/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/rfnet_algorithm.yaml)

```python
def __init__(self, **kwargs):
  self.model_path = kwargs.get("model_path")
```

3. Return the results

 ```return seen_image, unseen_image ```

#### 3.1.3 Interface Description

```yaml
- type: "unseen_sample_recognition"
# name of python module; string type;
  name: "UnseenSampleRecognitionByScene"
# the url address of python module; string type;
  url: "./examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/unseen_sample_recognition_by_scene.py"
# hyperparameters configuration for the python module; list type;
  hyperparameters:
    - model_path:
      values:
        - "./examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/models/Epochofprose17.pth"
```



### 3.2 Usecase

#### 3.2.1 Calling datasets

The current `testenv.yaml` file is as follows

```yaml
testenv:
  # dataset configuration
  dataset:
    # the url address of train dataset index; string type;
    train_url: "./ianvs/dataset/curb-detection/train_data/index.txt"
    # the url address of test dataset index; string type;
    test_url: "./ianvs/dataset/curb-detection/test_data/index.txt"

  # model eval configuration of incremental learning;
  model_eval:
    # metric used for model evaluation
    model_metric:
      # metric name; string type;
      name: "accuracy"
      # the url address of python file
      url: "./examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testenv/accuracy.py"

    # condition of triggering inference model to update
    # threshold of the condition; types are float/int
    threshold: 0
    # operator of the condition; string type;
    # values are ">=", ">", "<=", "<" and "=";
    operator: "<"

  # metrics configuration for test case's evaluation; list type;
  metrics:
      # metric name; string type;
    - name: "accuracy"
      # the url address of python file
      url: "./examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testenv/accuracy.py"
    - name: "samples_transfer_ratio"

```



#### 3.2.2 User flow chart

<img src="images/User.png" alt="User" style="zoom:48%;" />

Usage Process
1. View the dataset
2. download the edge cloud synergy dataset
3. Package your algorithm into Estimator
4. Call the Benchmark module, and the Benchmark module calls the developer's encapsulated Estimator
5. Run and view the results
6. Update and view the local edge cloud collaborative AI ranking



## 4 datasets

Two datasets, SYNTHIA, and cityscape, were selected for this project. Because SYNTHIA is often easier to obtain than the real urban road dataset as simulated by the simulator in a real research environment, it is treated as known task data for model pre-training, while the real urban landscape image acquisition requires more resources and is more difficult to obtain, so it is treated as unknown task data.

You can check [here](https://github.com/Frank-lilinjie/ianvs/tree/main/docs/proposals/algorithms/lifelong-learning/Additional-documentation/curb_detetion_datasets.md) for more details about curb_detetion_datasets.

## 5 Design Details

It is mainly about the algorithm of the training business model of the system, logical structure of task definition part, introduction of urban landscape dataset, neural network structure, and algorithm principle of unknown task recognition algorithm.

### 5.1 Model of the task definition section

This part of the training model algorithm uses the RFNet method mentioned in the RAL2020 paper **Real-Time Fusion Network for RGB-D Semantic Segmentation Incorporating Unexpected Obstacle Detection for Road-Driving Images** to train the model.

The entire network architecture of RFNet is shown in Fig. In the encoder part of the architecture, we design two independent branches to extract features for RGB and depth images separately RGB branch is the main branch, and the Depth branch is the subordinate branch. In both branches, we choose ResNet18 [30] as the backbone to extract features from inputs because ResNet-18 has moderate depth and residual structure, and its small operation footprint is compatible with the real-time operation. After each layer of ResNet-18, the output features from the Depth branch are fused to the RGB branch after the Attention Feature Complementary (AFC) module. The spatial pyramid pooling (SPP) block gathers the fused RGB-D features from two branches and produces feature maps with multi-scale information. Finally, referring to SwiftNet, we design the efﬁcient upsampling modules to restore the resolution of these feature maps with skip connections from the RGB branch.

<center>
  <img src="images/Overview_of_RFNet.png" style="zoom: 33%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
  </dir>
</center>

Fig. shows some examples from the validation set of Cityscapes and Lost and Found, which demonstrates the excellent segmentation accuracy of our RFNet in various scenarios with or without small obstacles.

<center>
  <img src="images/example.png" style="zoom: 33%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
  </dir>
</center>



You can check [RFnet](https://github.com/AHupuJR/RFNet) for more details.

### 5.2 Task Definition

The task definition is mainly accomplished by dividing the data into multiple parts with various classifications, and each part of the data is trained with a model to obtain multiple models.

For this project, we use two datasets to train the models separately: cityscape and SYNTHIA-RAND-CITYSCAPES, where the cityscape is the traveling camera data and SYNTHIA-RAND-CITYSCAPES is the simulation data.

#### 5.2.1 workflow

<center>
  <img src="images/task_definition.png" style="zoom: 33%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
  task definition
  </dir>
</center>



### 5.3 Unknown Task Recognition

This project aims to reproduce the CVPR2021 paper "Learning placeholders for open-set recognition" in a defect detection scenario.

#### 5.3.1 Workflow

The following is the workflow of the unknown task identification module. When faced with an inference task, the unknown task identification algorithm can give a timely indication of which data are known and which are unknown in the data set.

<center>
  <img src="images/unknow.png" style="zoom: 50%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
  </dir>
</center>



#### 5.3.2 Neural network structure

For the model network structure of this project, WideResnet network + placeholder is selected, and the following is the schematic diagram of model training. As shown in the figure, in the training phase, we use real camera acquisition data and simulator data as known class training data set, and train the model after WideResnet+Proser algorithm.

<center>
  <img src="images/Training.png" style="zoom: 50%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
    Training processing
  </dir>
</center>

In the testing phase, the test set consists of a combination of known and unknown class data, where the known class data are camera capture images and simulator images, while the unknown class data are natural forest images and cartoon images. After the model recognition, the data will be distinguished into known and unknown classes, where the known class data will be further classified according to the training categories.

<center>
  <img src="images/test.png" style="zoom: 50%;" />
  <br>
  <dir style="color:orange; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              color: #999;
              padding: 2px;">
    Inference processing
  </dir>
</center>



WideResnet source code: [WideResnet](https://github.com/szagoruyko/wide-residual-networks)

PROSER Algorithm principle: [PROSER](https://github.com/Frank-lilinjie/ianvs/tree/main/docs/proposals/algorithms/lifelong-learning/Additional-documentation/Open-set_recognition_Learning_Placeholders.md)

You can check the test report in [here](https://github.com/Frank-lilinjie/ianvs/tree/main/docs/proposals/algorithms/lifelong-learning/Additional-documentation/Testing_Open-set_recognition_in_Curb-detection_datasets.md)



## Roadmap

The roadmap would be as follows

### July

- Select image dataset；
- Complete the design of the task definition；

### August

- Completion of the task definition algorithm；
- Reproduction of unknown task recognition algorithms；
- Designing classifiers with the algorithm of the paper；

### September

- Improved accuracy of unknown task recognition algorithm to 90%；
- Merge into the lib of the Ianvs module；
- Intermodulation with other related projects；
=======
version https://git-lfs.github.com/spec/v1
oid sha256:812fa651e099682fc3e25dec2c512a8168ca9388066e84af6598f04aae12c84c
size 15387
>>>>>>> 9676c3e (ya toh aar ya toh par)
