# Testing incremental learning in industrial defect detection

## About Industrial Defect Detection 

In recent years, the manufacturing process is moving towards a higher degree of automation and improved manufacturing efficiency. During this development, smart manufacturing increasingly employs computing technologies, for example, with a higher degree of automation, there is also a higher risk in product defects; thus, a number of machine learning models have been developed to detect defectives in the manufacturing process.  

Defects are an unwanted thing in manufacturing industry. There are many types of defect in manufacturing like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc. For removing this defective product all industry have their defect detection department. But the main problem is this inspection process is carried out manually. It is a very time-consuming process and due to human accuracy, this is not 100\% accurate. This can because of the rejection of the whole order. So it creates a big loss in the company.

## About Dataset

The printed circuit board (PCB) industry is not different. Surface-mount technology (SMT) is a technology that automates PCB production in which components are mounted or placed onto the surface of printed circuit boards. Solder paste printing (SPP) is the most delicate stage in SMT. It prints solder paste on the pads of an electronic circuit panel. Thus, SPP is followed by a solder paste inspection (SPI) stage to detect defects. SPI scans the printed circuit board for missing/less paste, bridging between pads, miss alignments, and so forth. Boards with anomaly must be detected, and boards in good condition should not be disposed of. Thus SPI requires high precision and a high recall. 

As an example in this document, we are using [the PCB-AoI dataset](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) released by KubeEdge SIG AI members on Kaggle. See [this link](../scenarios/industrial-defect-detection/pcb-aoi.md) for more information of this dataset. Below also shows two example figures in the dataset. 

![](images/PCB-AoI_example.png)


## About Incremental Learning
Traditionally, the data is collected manually and periodically retrained on the cloud to improve the model effect. However, data is continuously generated on the edge side. Traditional method wastes a lot of human resources, and the model update frequency is slow.  

Incremental learning allows users to continuously monitor the newly generated data and by configuring some triggering rules to determine whether to start training, evaluation, and deployment automatically, and continuously improve the model performance.

Its goals include:
* Automatically retrains, evaluates, and updates models based on the data generated at the edge.
* Support time trigger, sample size trigger, and precision-based trigger.
* Support manual triggering of training, evaluation, and model update.
* Support hard sample discovering of unlabeled data,  for reducing the manual labeling workload. 


## Benchmark Setting

Key settings of the test environment to incremental learning are as follows: 
``` yaml
# testenv.yaml
testenv:
  dataset:
    train_ratio: 0.8
  model_eval:
    model_metric:
      name: "f1_score"
    threshold: 0
    operator: ">="
  metrics:
    - name: "f1_score"
  incremental_rounds: 2
```

Key settings of the algorithm to incremental learning are as follows: 
```yaml
# algorithm.yaml
algorithm:
  paradigm: "incrementallearning"
  dataset_train_ratio: 0.8
  modules:
    - kind: "basemodel"
      name: "estimator"
      hyperparameters:
        - momentum:
            values:
              - 0.8
              - 0.6
        - other_hyperparameters:
            values:
              learning_rate: 0.1

```
<!-- momentum: 0.9 -->

## Benchmark Result 

We release the leaderboard [here](../leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-single-task-learning.md).