# Testing single task learning in industrial defect detection

## About Industrial Defect Detection 

In recent years, the manufacturing process is moving towards a higher degree of automation and improved manufacturing efficiency. During this development, smart manufacturing increasingly employs computing technologies, for example, with a higher degree of automation, there is also a higher risk in product defects; thus, a number of machine learning models have been developed to detect defectives in the manufacturing process.  

Defects are an unwanted thing in manufacturing industry. There are many types of defect in manufacturing like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc. For removing this defective product all industry have their defect detection department. But the main problem is this inspection process is carried out manually. It is a very time-consuming process and due to human accuracy, this is not 100\% accurate. This can because of the rejection of the whole order. So it creates a big loss in the company.


## About Dataset

The printed circuit board (PCB) industry is not different. Surface-mount technology (SMT) is a technology that automates PCB production in which components are mounted or placed onto the surface of printed circuit boards. Solder paste printing (SPP) is the most delicate stage in SMT. It prints solder paste on the pads of an electronic circuit panel. Thus, SPP is followed by a solder paste inspection (SPI) stage to detect defects. SPI scans the printed circuit board for missing/less paste, bridging between pads, miss alignments, and so forth. Boards with anomaly must be detected, and boards in good condition should not be disposed of. Thus SPI requires high precision and a high recall. 

As an example in this document, we are using [the PCB-AoI dataset](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) released by KubeEdge SIG AI members on Kaggle. See [this link](../scenarios/industrial-defect-detection/pcb-aoi.md) for more information of this dataset. Below also shows two example figures in the dataset. 

![](images/PCB-AoI_example.png)

## About Single Task Learning
It is a traditional learning pooling all data together to train a single model. It typically includes a specialist model laser-focused on a single task and requires large amounts of task-specific labeled data, which is not always available on early stage of a distributed synergy AI project. 

As for the base model of single task learning, in this report we are using FPN_TensorFlow. It is a tensorflow re-implementation of Feature Pyramid Networks for Object Detection, which is based on Faster-RCNN. More detailedly, feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. Researchers have exploited the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. The architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, the method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-task entries including those from the COCO 2016 challenge winners. In addition, FPN can run at 5 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. 

The ``FPN_TensorFlow`` is also open sourced and completed by YangXue and YangJirui. For those interested in details of ``FPN_TensorFlow``, an example implementation is available [here](https://github.com/DetectionTeamUCAS/FPN_Tensorflow) and is extended with the Ianvs algorithm inferface [here](https://github.com/kubeedge-sedna/FPN_Tensorflow).

## Benchmark Setting
Key settings of the test environment to single task learning are as follows: 
``` yaml
# testenv.yaml
testenv:
  # dataset configuration
  dataset:
    train_ratio: 0.8
  # metrics configuration for test case's evaluation; list type;
  metrics:
    - name: "f1_score"
```

Key settings of the algorithm to single learning are as follows: 
```yaml
# algorithm.yaml
algorithm:
  # paradigm name; string type;
  paradigm: "singletasklearning"
  # algorithm module configuration in the paradigm; list type;
  modules:
      # kind of algorithm module; string type;
    - kind: "basemodel"
      # name of python module; string type;
      name: "estimator"

      # hyperparameters configuration for the python module; list type;
      hyperparameters:
          # name of the hyperparameter; string type;
        - momentum:
            # values of the hyperparameter; list type;
            values:
              - 0.7
              - 0.5
          # hyperparameters configuration files; dictionary type;
        - other_hyperparameters:
            values:
              learning_rate: 0.1

```


## Benchmark Result 

We release the leaderboard [here](../leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-single-task-learning.md).