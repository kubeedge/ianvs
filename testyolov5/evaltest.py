from tkinter import Widget
import torch
import numpy as np
# # from yolo_model.eval import performance_evaluation_map_teacher
# model = torch.hub.load('yolov5','custom', path ='/mnt/disk/shifan/ianvs/yolov5s.pt', source='local', device='2')
# img = "/mnt/disk/detection_ped_car/val/images/b1c66a42-6f7d68ca.jpg"
# results = model(img)
# results = np.array(results.pandas().xywhn[0])
# # print(dir(results.pandas()))
# print(results)

# inference_boxs = []

# print(results)
# print("1111111111111111111111111111")

# box_pred_list = results[:, 0:4]
# print(box_pred_list)
# print("1111111111111111111111111111")
# confidence_list = results[:, 4]
# print(confidence_list)
# print("1111111111111111111111111111")
# class_pred_list = results[:, 5]
# print(class_pred_list)
# print("1111111111111111111111111111")


selected_model_path = "/mnt/disk/shifan/ianvs/yolo_model/"
weight_list = ['all.pt', 'bdd.pt', 'traffic_0.pt', 'bdd_street.pt', 'bdd_clear.pt', 'bdd_daytime.pt',
                      'bdd_highway.pt', 'traffic_2.pt', 'bdd_overcast.pt', 'bdd_residential.pt', 'traffic_1.pt', 
                      'bdd_snowy.pt', 'bdd_rainy.pt', 'bdd_night.pt', 'soda.pt', 'bdd_cloudy.pt', 'bdd_cloudy_night.pt',
                      'bdd_highway_residential.pt', 'bdd_snowy_rainy.pt', 'soda_t1.pt']
model = torch.hub.load('ultralytics/yolov5', 'custom', path=selected_model_path+weight_list[0])
 # vertify weather the signature is useful