# Benchmarking of pedestrian tracking

Person Re-ID based on Cloud-edge collaborative architecture is an example feature that [Sedna](https://github.com/kubeedge/sedna) supports, which can continuously recognize, track and search the target person provided by the user in the source video, and push the video containing the search results to the streaming media server. This guide provides users with the actual performance test of the AI algorithm as a reference. To simulate the workflow in Sedna, the benchmarking process consists of a tracking job and a Re-ID job. The MOT17 dataset is used as the input for both tracking and Re-ID. After running benchmarking jobs, a report will be generated.

With Ianvs installed and related environment prepared, users is then able to run the benchmarking process using the following steps. If you haven't installed Ianvs, please refer to [how-to-install-ianvs](../../docs/guides/how-to-install-ianvs.md).

## Prerequisites

- CUDA >= 10.0

To setup the environment, run the following commands:
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
python setup.py develop
pip install cython
pip install cython_bbox
pip install pycocotools
pip install scikit-learn seaborn fpdf
cd <Ianvs_HOME>
mkdir dataset initial_model
```

## Step 1. Run tracking benchmark job

Download [MOT17](https://motchallenge.net/) and put it under <Ianvs_HOME>/dataset in the following structure:

```
dataset
   |------mot17
   |        |------train
   |        |------test
```

Download [convert_mot17_to_coco.py](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py)

Then, you need to turn the dataset to COCO format:

```shell
sed -i 's/datasets\/mot/dataset\/mot17/g' convert_mot17_to_coco.py
python convert_mot17_to_coco.py
```

Next, download pretrained model via [[google]](https://drive.google.com/file/d/1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob/view?usp=sharing) or [[baidu(code:eeo8)]](https://pan.baidu.com/s/1W5eRBnxc4x9V8gm7dgdEYg) and put it under <Ianvs_HOME>/initial_model

We are now ready to run the ianvs for benchmarking pedestrian tracking on the MOT17 dataset.

```python
ianvs -f ./examples/pedestrian_tracking/multiedge_inference_bench/tracking_job.yaml
```

The benchmarking process takes a few minutes and varies depending on devices.
Finally, the user can check the result of benchmarking on the console and also in the output path( /ianvs/multiedge_inference_bench/workspace) defined in the benchmarking config file ( tracking_job.yaml). 
The final output might look like this:

|rank  |algorithm                | mota | f1_score  |motp  |recall  |idf1  |precision  |paradigm            |basemodel  |batch_size  |time                     |url                                                                                                                             |
|:----:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------------------:|:---------:|:--------:|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
|1     |tracking  | 0.7644 | 0.8772 | 0.1554 | 0.8305 | 0.7949 | 0.9294 |multiedgeinference  | ByteTrack       | 1           | 2022-09-16 11:40:15     |/ianvs/multiedge_inference_bench/workspace/tracking_job/tracking/5886b6d8-35b3-11ed-b2cf-fc3497a39dd9 |
|2     |tracking  | 0.7644 | 0.8772 | 0.1554 | 0.8305 | 0.7949 | 0.9294 |multiedgeinference  | ByteTrack       | 1           | 2022-09-16 10:16:06     |/ianvs/multiedge_inference_bench/workspace/tracking_job/tracking/846c71fe-35a7-11ed-981c-fc3497a39dd9 |

## Step 2. Run ReID benchmark job

Download [mot2reid.py](https://github.com/open-mmlab/mmtracking/blob/master/tools/convert_datasets/mot/mot2reid.py)

Then you need to generate the dataset for ReID job:

```shell
cd <Ianvs_HOME>
python mot2reid.py -i ./dataset/mot17/ -o ./dataset/mot17/reid
mv dataset/mot17/reid/meta/* dataset/mot17/reid/
sed -i 's/^/imgs\//g' dataset/mot17/reid/train.txt
sort -u -k2,2 dataset/mot17/reid/train.txt > dataset/mot17/reid/test.txt
```

Next, download pretrained model via [[google]](https://drive.google.com/drive/folders/1P_1nsTirOQ_8OZU0rgEx9eH1M34v5S0v?usp=sharing) and put it under <Ianvs_HOME>/initial_model

We are now ready to run the ianvs for benchmarking pedestrian re-identification on the MOT17 dataset.

```python
ianvs -f ./examples/pedestrian_tracking/multiedge_inference_bench/reid_job.yaml
```

The benchmarking process takes a few minutes and varies depending on devices.
Finally, the user can check the result of benchmarking on the console and also in the output path( /ianvs/multiedge_inference_bench/workspace) defined in the benchmarking config file ( reid_job.yaml). 
The final output might look like this:

|rank  |algorithm                |rank_1  |mAP  |cmc  |rank_2  |rank_5  |paradigm            |basemodel  |batch_size  |time                     |url                                                                                                                             |
|:----:|:-----------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:------------------:|:---------:|:--------:|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
|1     |feature extraction  | 0.3340 | 0.2487 | examples/pedestrian_tracking/cmc/20220916102704.png | 0.6681 |  1.0   | multiedgeinference |M3L       | 32           | 2022-09-16 10:27:04     |/ianvs/multiedge_inference_bench/workspace/reid_job/feature extraction/a8e142ac-35a8-11ed-8111-fc3497a39dd9 |
|2     |feature extraction  | 0.3354 | 0.2487 | examples/pedestrian_tracking/cmc/20220916110052.png  | 0.6667 |  1.0   | multiedgeinference | M3L       | 32           | 2022-09-16 11:00:52     |/ianvs/multiedge_inference_bench/workspace/reid_job/feature extraction/64680d5e-35ad-11ed-8793-fc3497a39dd9 |

## Step 3. Generate test report

```shell
python ./examples/pedestrian_tracking/generate_reports.py \
-t ./examples/pedestrian_tracking/multiedge_inference_bench/tracking_job.yaml \
-r ./examples/pedestrian_tracking/multiedge_inference_bench/reid_job.yaml
```

Finally, the report is generated under <Ianvs_HOME>/examples/multiedge_inference/reports. You can also check the sample report under the current directory.

## What is next

If the reader is ready to explore more on this example, e.g. upload custom algorithms, the following links might help:

[ByteTrack](https://github.com/ifzhang/ByteTrack)

[M3L](https://github.com/HeliosZhao/M3L)

[Sedna's ReID feature](https://github.com/kubeedge/sedna/tree/main/examples/multiedgeinference/pedestrian_tracking)

[How to test algorithms](../../docs/guides/how-to-test-algorithms.md)

[How to contribute algorithms](../../docs/guides/how-to-contribute-algorithms.md)

[How to contribute test environments](../../docs/guides/how-to-contribute-test-environments.md)

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue. 

Enjoy your journey on Ianvs!

