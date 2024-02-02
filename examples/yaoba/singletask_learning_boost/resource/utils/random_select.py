import os
import random
from random import sample


def random_select(src_txt, num, out_dir, select_name, rest_name):
    fp = open(src_txt, mode="r", encoding='utf-8')
    results = fp.readlines()
    fp_select = open(os.path.join(out_dir, select_name), mode="a+", encoding='utf-8')
    fp_rest = open(os.path.join(out_dir, rest_name), mode="a+", encoding='utf-8')
    selected_items = sample(results, num)
    for result in results:
        if result in selected_items:
            fp_select.write(result)
        else:
            fp_rest.write(result)


if __name__ == '__main__':
    random.seed(5)
    random_select("/custom_code/instance_based/txt/merged_part/NG.txt",
                  139,
                  "/home/wjj/wjj/Public/code/huawei/custom_code/instance_based/txt/merged_part",
                  select_name="NG_test.txt",
                  rest_name="NG_labeled.txt")
