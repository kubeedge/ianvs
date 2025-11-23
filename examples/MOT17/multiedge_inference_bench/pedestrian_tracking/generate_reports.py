<<<<<<< HEAD
# Modified Copyright 2022 The KubeEdge Authors.
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

import argparse
import datetime
import pandas as pd
from pathlib import Path
from fpdf import FPDF
from core.common import utils


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, 'Test report', 0, 0, 'R')
        self.ln(20)

    def page_body(self, results):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        tracking_result = results[0]
        reid_result = results[1]
        self.text(20, 25, "Tracking")
        self.text(20, 35, "Time: " + str(tracking_result["time"]))
        self.text(20, 45, "Paradigm: " + str(tracking_result["paradigm"]))
        self.text(20, 55, "Algorithm: " + str(tracking_result["algorithm"]))
        self.text(20, 65, "Basemodel: " + str(tracking_result["basemodel"]))
        self.text(20, 75, "Batch size: " + str(tracking_result["batch_size"]))
        self.text(20, 85, "Metrics")
        self.text(20, 95, "Precision: " + "{:.2%}".format(tracking_result["precision"]))
        self.text(20, 105, "Recall: " + "{:.2%}".format(tracking_result["recall"]))
        self.text(20, 115, "F1 score: " + "{:.2%}".format(tracking_result["f1_score"]))
        self.text(20, 125, "MOTA: " + "{:.2%}".format(tracking_result["mota"]))
        self.text(20, 135, "MOTP: " + "{:.2%}".format(tracking_result["motp"]))
        self.text(20, 145, "ID f1 score: " + "{:.2%}".format(tracking_result["idf1"]))
        self.text(20, 165, "ReID")
        self.text(20, 175, "Time: " + str(reid_result["time"]))
        self.text(20, 185, "Paradigm: " + str(reid_result["paradigm"]))
        self.text(20, 195, "Algorithm: " + str(reid_result["algorithm"]))
        self.text(20, 205, "Basemodel: " + str(reid_result["basemodel"]))
        self.text(20, 215, "Batch size: " + str(reid_result["batch_size"]))
        self.text(20, 225, "Metrics")
        self.text(20, 235, "mAP: " + "{:.2%}".format(reid_result["mAP"]))
        self.text(20, 245, "Rank 1: " + "{:.2%}".format(reid_result["rank_1"]))
        self.text(20, 255, "Rank 2: " + "{:.2%}".format(reid_result["rank_2"]))
        self.text(20, 265, "Rank 5: " + "{:.2%}".format(reid_result["rank_5"]))
        self.image(reid_result["cmc"], 100, 185, 100)

    def print_page(self, results):
        # Generates the report
        self.add_page()
        self.page_body(results)

def main():
    try:
        parser = _generate_parser()
        args = parser.parse_args()
        tracking_config_file = args.tracking_benchmarking_config_file
        if not utils.is_local_file(tracking_config_file):
            raise SystemExit(f"not found benchmarking config({config_file}) file in local")

        tracking_config = utils.yaml2dict(args.tracking_benchmarking_config_file)
        tracking_rank = pd.read_csv(Path(tracking_config["benchmarkingjob"]["workspace"], tracking_config["benchmarkingjob"]["name"], "rank/all_rank.csv"), delim_whitespace=True)
        tracking_rank["time"] = pd.to_datetime(tracking_rank["time"])
        tracking_result = tracking_rank.sort_values(by="time", ascending=False).iloc[0]

        reid_config_file = args.reid_benchmarking_config_file
        if not utils.is_local_file(reid_config_file):
            raise SystemExit(f"not found benchmarking config({config_file}) file in local")
        reid_config = utils.yaml2dict(args.reid_benchmarking_config_file)
        reid_rank = pd.read_csv(Path(reid_config["benchmarkingjob"]["workspace"], reid_config["benchmarkingjob"]["name"], "rank/all_rank.csv"), delim_whitespace=True)
        reid_rank["time"] = pd.to_datetime(reid_rank["time"])
        reid_result = reid_rank.sort_values(by="time", ascending=False).iloc[0]
        pdf = PDF()
        pdf.print_page([tracking_result, reid_result])
        output_dir = Path("./examples/pedestrian_tracking/multiedge_inference_bench/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf.output(Path(output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"), "F")
    except Exception as err:
        raise Exception(f"test report generation runs failed, error: {err}.") from err

def _generate_parser():
    parser = argparse.ArgumentParser(description='Test Report Generation Tool')
    parser.add_argument("-t",
                        "--tracking_benchmarking_config_file",
                        nargs="?",
                        type=str,
                        help="the benchmarking config file must be yaml/yml file.")
    parser.add_argument("-r",
                        "--reid_benchmarking_config_file",
                        nargs="?",
                        type=str,
                        help="the benchmarking config file must be yaml/yml file.")
    return parser

if __name__ == '__main__':
    main()
=======
version https://git-lfs.github.com/spec/v1
oid sha256:1ce5d023aea6ad9c12b8fb839c043107fc36bfff67d6b1a1999924c4dc6a3f09
size 5522
>>>>>>> 9676c3e (ya toh aar ya toh par)
