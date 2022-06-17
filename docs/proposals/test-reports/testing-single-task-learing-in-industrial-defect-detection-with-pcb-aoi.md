# Testing single task learning in industrial defect detection

## About Industrial Defect Detection 

In recent years, the manufacturing process is moving towards a higher degree of automation and improved manufacturing efficiency. During this development, smart manufacturing increasingly employs computing technologies, for example, with a higher degree of automation, there is also a higher risk in product defects; thus, a number of machine learning models have been developed to detect defectives in the manufacturing process.  

Defects are an unwanted thing in manufacturing industry. There are many types of defect in manufacturing like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc. For removing this defective product all industry have their defect detection department. But the main problem is this inspection process is carried out manually. It is a very time-consuming process and due to human accuracy, this is not 100\% accurate. This can because of the rejection of the whole order. So it creates a big loss in the company.


## About Dataset

The printed circuit board (PCB) industry is not different. Surface-mount technology (SMT) is a technology that automates PCB production in which components are mounted or placed onto the surface of printed circuit boards. Solder paste printing (SPP) is the most delicate stage in SMT. It prints solder paste on the pads of an electronic circuit panel. Thus, SPP is followed by a solder paste inspection (SPI) stage to detect defects. SPI scans the printed circuit board for missing/less paste, bridging between pads, miss alignments, and so forth. Boards with anomaly must be detected, and boards in good condition should not be disposed of. Thus SPI requires high precision and a high recall. 

As an example in this document, we are using [the PCB-AoI dataset](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) released by KubeEdge SIG AI members on Kaggle. See [this link](../scenarios/industrial-defect-detection/pcb-aoi.md) for more information of this dataset. Below also shows two example figures in the dataset. 

![](images/PCB-AoI_example.png)

## About Single Task Learning

## Setting

## Result 

We release the leaderboard [here](../leaderboards/leaderboard-in-industrial-defect-detection-of-PCB-AoI/leaderboard-of-incremental-learning.md).