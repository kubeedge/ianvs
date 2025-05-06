# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["acc"]

def get_last_letter(input_string):
    if not input_string or not any(char.isalpha() for char in input_string):
        return None
    
    for char in reversed(input_string):
        if 'A' <= char <= 'D':
            return char
        
    return None


@ClassFactory.register(ClassType.GENERAL, alias="acc_model")
def acc_model(y_true, y_pred):
    original_preds = y_pred.copy()

    y_pred = [pred.split("||")[0] for pred in original_preds]
    y_true = [pred.split("||")[1] for pred in original_preds]
    y_location = [pred.split("||")[2] for pred in original_preds]
    y_source = [pred.split("||")[3] for pred in original_preds]

    real_y_pred = []
    real_y_true = []
    real_locations = []

    for i in range(len(y_pred)):
        if y_source[i] == "[model]":
            real_y_pred.append(get_last_letter(y_pred[i]))
            real_y_true.append(y_true[i])
            real_locations.append(y_location[i])

    same_elements = [get_last_letter(real_y_pred[i]) == real_y_true[i] for i in range(len(real_y_pred))]
    global_acc = sum(same_elements) / len(same_elements)

    province_acc = {}
    for i in range(len(real_y_pred)):
        province = real_locations[i]
        if province not in province_acc:
            province_acc[province] = {"correct": 0, "total": 0}
        
        province_acc[province]["total"] += 1
        if real_y_pred[i] == real_y_true[i]:
            province_acc[province]["correct"] += 1

    print("\n=== Accuracy Statistics ===")
    print(f"Global Accuracy: {global_acc:.4f}")
    print("\nProvince Accuracies:")
    for province, stats in province_acc.items():
        acc = stats["correct"] / stats["total"]
        print(f"{province}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    import json
    from datetime import datetime
    
    results = {
        "global_accuracy": global_acc,
        "province_accuracies": {
            province: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for province, stats in province_acc.items()
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("accuracy_results_model.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return global_acc


@ClassFactory.register(ClassType.GENERAL, alias="acc_global")
def acc_global(y_true, y_pred):
    original_preds = y_pred.copy()

    y_pred = [pred.split("||")[0] for pred in original_preds]
    y_true = [pred.split("||")[1] for pred in original_preds]
    y_location = [pred.split("||")[2] for pred in original_preds]
    y_source = [pred.split("||")[3] for pred in original_preds]

    real_y_pred = []
    real_y_true = []
    real_locations = []

    for i in range(len(y_pred)):
        if y_source[i] == "[global]":
            real_y_pred.append(get_last_letter(y_pred[i]))
            real_y_true.append(y_true[i])
            real_locations.append(y_location[i])

    same_elements = [get_last_letter(real_y_pred[i]) == real_y_true[i] for i in range(len(real_y_pred))]
    global_acc = sum(same_elements) / len(same_elements)

    province_acc = {}
    for i in range(len(real_y_pred)):
        province = real_locations[i]
        if province not in province_acc:
            province_acc[province] = {"correct": 0, "total": 0}
        
        province_acc[province]["total"] += 1
        if real_y_pred[i] == real_y_true[i]:
            province_acc[province]["correct"] += 1

    print("\n=== Accuracy Statistics ===")
    print(f"Global Accuracy: {global_acc:.4f}")
    print("\nProvince Accuracies:")
    for province, stats in province_acc.items():
        acc = stats["correct"] / stats["total"]
        print(f"{province}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    import json
    from datetime import datetime
    
    results = {
        "global_accuracy": global_acc,
        "province_accuracies": {
            province: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for province, stats in province_acc.items()
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("accuracy_results_global.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return global_acc


@ClassFactory.register(ClassType.GENERAL, alias="acc_local")
def acc_local(y_true, y_pred):
    original_preds = y_pred.copy()

    y_pred = [pred.split("||")[0] for pred in original_preds]
    y_true = [pred.split("||")[1] for pred in original_preds]
    y_location = [pred.split("||")[2] for pred in original_preds]
    y_source = [pred.split("||")[3] for pred in original_preds]

    real_y_pred = []
    real_y_true = []
    real_locations = []

    for i in range(len(y_pred)):
        if y_source[i] == "[local]":
            real_y_pred.append(get_last_letter(y_pred[i]))
            real_y_true.append(y_true[i])
            real_locations.append(y_location[i])

    same_elements = [get_last_letter(real_y_pred[i]) == real_y_true[i] for i in range(len(real_y_pred))]
    global_acc = sum(same_elements) / len(same_elements)

    province_acc = {}
    for i in range(len(real_y_pred)):
        province = real_locations[i]
        if province not in province_acc:
            province_acc[province] = {"correct": 0, "total": 0}
        
        province_acc[province]["total"] += 1
        if real_y_pred[i] == real_y_true[i]:
            province_acc[province]["correct"] += 1

    print("\n=== Accuracy Statistics ===")
    print(f"Global Accuracy: {global_acc:.4f}")
    print("\nProvince Accuracies:")
    for province, stats in province_acc.items():
        acc = stats["correct"] / stats["total"]
        print(f"{province}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    import json
    from datetime import datetime
    
    results = {
        "global_accuracy": global_acc,
        "province_accuracies": {
            province: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for province, stats in province_acc.items()
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("accuracy_results_local.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return global_acc


@ClassFactory.register(ClassType.GENERAL, alias="acc_other")
def acc_other(y_true, y_pred):
    original_preds = y_pred.copy()

    y_pred = [pred.split("||")[0] for pred in original_preds]
    y_true = [pred.split("||")[1] for pred in original_preds]
    y_location = [pred.split("||")[2] for pred in original_preds]
    y_source = [pred.split("||")[3] for pred in original_preds]

    real_y_pred = []
    real_y_true = []
    real_locations = []

    for i in range(len(y_pred)):
        if y_source[i] == "[other]":
            real_y_pred.append(get_last_letter(y_pred[i]))
            real_y_true.append(y_true[i])
            real_locations.append(y_location[i])

    same_elements = [get_last_letter(real_y_pred[i]) == real_y_true[i] for i in range(len(real_y_pred))]
    global_acc = sum(same_elements) / len(same_elements)

    province_acc = {}
    for i in range(len(real_y_pred)):
        province = real_locations[i]
        if province not in province_acc:
            province_acc[province] = {"correct": 0, "total": 0}
        
        province_acc[province]["total"] += 1
        if real_y_pred[i] == real_y_true[i]:
            province_acc[province]["correct"] += 1

    print("\n=== Accuracy Statistics ===")
    print(f"Global Accuracy: {global_acc:.4f}")
    print("\nProvince Accuracies:")
    for province, stats in province_acc.items():
        acc = stats["correct"] / stats["total"]
        print(f"{province}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    import json
    from datetime import datetime
    
    results = {
        "global_accuracy": global_acc,
        "province_accuracies": {
            province: {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for province, stats in province_acc.items()
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("accuracy_results_other.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    return global_acc