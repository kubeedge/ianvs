# Ianvs Examples

For status meanings, badge definitions, and broken-status subtypes, see [`status_directions.md`](../docs/example_validator/status_directions.md).

**Last T2/T3 Validation Time:** <img alt="validated at" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fsummary.json&amp;query=%24.validated_at_display&amp;label=validated+at&amp;cacheSeconds=300">

## Example Classification Matrix

<table>
  <thead>
    <tr>
      <th>Example</th>
      <th>Benchmark Unit</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Cloud_Robotics</td>
      <td><a href="./Cloud_Robotics/cloud-edge-collaborative-inference_bench/perception-reasoning">perception_reasoning_test</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FCloud_Robotics.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./Cloud_Robotics/singletask_learning_bench/Semantic_Segmentation">rfnet_singletask_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FCloud_Robotics.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>GovDoc2Poster</td>
      <td><a href="./GovDoc2Poster/singletask_learning_bench">government_poster_agent</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FGovDoc2Poster.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">MOT17</td>
      <td><a href="./MOT17/multiedge_inference_bench/pedestrian_tracking">feature_extraction</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FMOT17.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./MOT17/multiedge_inference_bench/pedestrian_tracking">tracking</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FMOT17.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>PIPL</td>
      <td><a href="./PIPL/edge-cloud_collaborative_learning_bench">privacy_preserving_llm_collaboration</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FPIPL.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>RoboDK Palletizing</td>
      <td><a href="./RoboDK%20Palletizing/singletask_learning_bench">palletizing_detection</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FRoboDK_Palletizing.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>TAB</td>
      <td><a href="./TAB/cloud_edge_collaborative_inference_bench">privacy_aware_query_routing</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FTAB.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>aoa</td>
      <td><a href="./aoa">execution_yaml_pending</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Faoa.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>bdd</td>
      <td><a href="./bdd/lifelong_learning_bench/curb-detection">yolo_lifelong_learning_five_model</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fbdd.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="5">cifar100</td>
      <td><a href="./cifar100/fci_ssl/fed_ci_match_v2">fci_ssl_test</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcifar100.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cifar100/federated_learning/fedavg">fedavg_test</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcifar100.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cifar100/federated_class_incremental_learning/fedavg">federated_class_incremental_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcifar100.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cifar100/fci_ssl/fed_ci_match">fedi_carl</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcifar100.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cifar100/fci_ssl/glfc">glfc_match</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcifar100.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>cityscapes</td>
      <td><a href="./cityscapes/singletask_learning_bench/semantic-segmentation">rfnet_singletask_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="6">cityscapes-synthia</td>
      <td><a href="./cityscapes-synthia/lifelong_learning_bench/curb-detection">rfnet_lifelong_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cityscapes-synthia/lifelong_learning_bench/semantic-segmentation">rfnet_lifelong_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cityscapes-synthia/scene-based-unknown-task-recognition/curb-detection">rfnet_lifelong_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cityscapes-synthia/lifelong_learning_bench/semantic-segmentation">rfnet_lifelong_learning_full_test</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cityscapes-synthia/lifelong_learning_bench/semantic-segmentation">rfnet_lifelong_learning_small_test</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cityscapes-synthia/lifelong_learning_bench/semantic-segmentation">rfnet_lifelong_learning_travel_mode</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>cloud-edge-collaborative-inference-for-llm</td>
      <td><a href="./cloud-edge-collaborative-inference-for-llm">query_routing</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcloud-edge-collaborative-inference-for-llm.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">cloud-edge-speculative-decoding-benchmark</td>
      <td><a href="./cloud-edge-speculative-decoding-benchmark">speculative_decoding</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcloud-edge-speculative-decoding-benchmark.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./cloud-edge-speculative-decoding-benchmark">speculative_decoding_block</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcloud-edge-speculative-decoding-benchmark.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>cloud_VLA_finetune</td>
      <td><a href="./cloud_VLA_finetune/singletask_learning_bench">vla_data_selection</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fcloud_VLA_finetune.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>federated-llm</td>
      <td><a href="./federated-llm/fedllm-peft">fedavgm_peft</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Ffederated-llm.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">government</td>
      <td><a href="./government/singletask_learning_bench/objective">objective_political_question_answering</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fgovernment.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./government/singletask_learning_bench/subjective">subjective_political_question_answering</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fgovernment.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>government_rag</td>
      <td><a href="./government_rag/singletask_learning_bench">rag_singletask_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fgovernment_rag.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">imagenet</td>
      <td><a href="./imagenet/multiedge_inference_bench">automatic_classification</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fimagenet.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./imagenet/multiedge_inference_bench">manual_classification</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fimagenet.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">industrialEI</td>
      <td><a href="./industrialEI/single_task_learning_bench/deformable_component_manipulation">deformable_component_assembly</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FindustrialEI.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./industrialEI/pose-estimation-llio">llio_pose_estimation</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2FindustrialEI.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>llm-agent</td>
      <td><a href="./llm-agent/singletask_learning_bench">llm_agent</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fllm-agent.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">llm-edge-benchmark-suite</td>
      <td><a href="./llm-edge-benchmark-suite/single_task_bench">llama_cpp_single_task</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fllm-edge-benchmark-suite.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./llm-edge-benchmark-suite/single_task_bench_with_compression">llama_cpp_single_task_with_compression</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fllm-edge-benchmark-suite.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>llm_simple_qa</td>
      <td><a href="./llm_simple_qa">simple_qa_singletask_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fllm_simple_qa.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">pcb-aoi</td>
      <td><a href="./pcb-aoi/incremental_learning_bench/fault_detection">fpn_incremental_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fpcb-aoi.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./pcb-aoi/singletask_learning_bench/fault_detection">fpn_singletask_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fpcb-aoi.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>phys_scene_gen</td>
      <td><a href="./phys_scene_gen/singletask_learning_bench">physical_scene_generation</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fphys_scene_gen.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">robot</td>
      <td><a href="./robot/lifelong_learning_bench/semantic-segmentation">rfnet_lifelong_learning_simple</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Frobot.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./robot/lifelong_learning_bench/semantic-segmentation">sam_rfnet_lifelong_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Frobot.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td>robot-cityscapes-synthia</td>
      <td><a href="./robot-cityscapes-synthia/lifelong_learning_bench/semantic-segmentation">erfnet_lifelong_learning</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Frobot-cityscapes-synthia.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">smart_coding</td>
      <td><a href="./smart_coding/smart_coding_learning_bench/comment">code_comment_generation</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fsmart_coding.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./smart_coding/smart_coding_learning_bench/issue">issue_response_generation</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fsmart_coding.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td rowspan="2">yaoba</td>
      <td><a href="./yaoba/singletask_learning_boost">mmlab_model_boost</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fyaoba.json&amp;cacheSeconds=300"></td>
    </tr>
    <tr>
      <td><a href="./yaoba/singletask_learning_yolox_tta">mmlab_model_yolox_tta</a></td>
      <td><img alt="status" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkubeedge%2Fianvs%2Fci-managed%2Fexample-health-status%2F.github%2Fexample-status%2Fyaoba.json&amp;cacheSeconds=300"></td>
    </tr>
  </tbody>
</table>
