# Semantic Segmentation based on Lifelong Learning of Ianvs

# About Task1

### Datasets

You can get details about the cityscapes dataset by looking at the following website [cityscapes](https://frank-lilinjie.github.io/)， including **Dataset overview**, **Lifelong learning algorithm overview**, **Download link**, and **benchmark** of semantic segmentation algorithms in cityscapes.



# About Task2

### Models Evaluate

Evaluate the performance on the single tasks, i.e., subsets of classes of the Cityscapes dataset, and also on the cross-task metrics over several subsets. The mIoU Task 1∪2∪3 metric represents the usual mIoU over 19 classes defined in the Cityscapes dataset.

Model | mIoU Task 1 | mIoU Task2 | mIoU Task 1∪2 | mIoU Task 3 | mIoU Task 1∪2∪3 
--- | --- | --- | --- |--- | --- 
Erfnet | 84.5 | 57.8 | 64.0 | 68.4 | 62.2 
Baseline | 88.1 | 65.6 | 70.8 | 63.8 | 64.4 

## Evaluation
You can evaluate the trained models with the following command:
```
python3 evaluate/evaluate_erfnet.py --load_model_name erfnet_incremental_set123 \
                                    --train_set 123 \
                                    --weights_epoch 199 \
                                    --task_to_val 123
```

## Lifelong learning Training

Train the first network with the first training data subset:
```
python3 train_erfnet_static.py --model_name erfnet_incremental_set1 \
                               --train_set 1 \
                               --num_epochs 200 \
                               --validate \
                               --city
```

Train the second network with the first network as a teacher on the second data subset:
```
python3 train_erfnet_incremental.py --model_name erfnet_incremental_set12 \
                                    --train_set 2 --num_epochs 200 \
                                    --validate \
                                    --teachers erfnet_static_set1 199 1 
```

Train the third network with the second network as a teacher on the third data subset:
```
python3 train_erfnet_incremental.py --model_name erfnet_incremental_set123 \
                                    --train_set 3 \
                                    --num_epochs 200 \
                                    --validate \
                                    --teachers erfnet_incremental_set12 199 2
```

### Furthermore

1. You can go further and put in new datasets and train the model.
2. You can put the [Unknown task recognition](https://github.com/kubeedge/ianvs/blob/feature-lifelong-n/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/Readme.md), in the lifelonglearning pipeline.

## Baseline Training

For training of our baseline model

```
python3 train_erfnet_static.py --model_name erfnet_static_set123 \
                               --train_set 123 \
                               --num_epochs 200
```

## Pre-trained models
You can download the pre-trained models here:

<a href="https://pan.baidu.com/s/1Y8BwOO29Ii5B90aeLm7E6g">erfnet_incremental_set1</a>access code:vdwo

<a href="https://pan.baidu.com/s/1KUAUccVCcLwycnLAmuWc8Q">erfnet_incremental_set12</a>access code:ult0

<a href="https://pan.baidu.com/s/1bdv7yMXiLXtimsikHWJCew">erfnet_incremental_set123</a>access code:fnaq

<a href="https://pan.baidu.com/s/1jdc8aZ4wSY3CGzt03Cdwjw">erfnet_static_set123</a>access code:fmen

and evaluate them with the commands above, and you should put the Pre-trained models in the **Checkpoints**.