import json
import numpy as np

np.set_printoptions(precision=4)
import copy

import torch

from dataloader.action_genome import AG, cuda_collate_fn
from dataloader.vidvrd import VidVRD
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.sttran import STTran

conf = Config()
for i in conf.args:
    print(i, ":", conf.args[i])

if conf.dataset == "ag":
    dataset_test = AG(
        mode="test",
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == "predcls" else True,
    )
elif conf.dataset == "vidvrd":
    dataset_test = VidVRD(
        mode="test",
        datasize=conf.datasize,
        data_path=conf.data_path,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
else:
    raise ValueError(f"{conf.dataset}")

dataloader = torch.utils.data.DataLoader(
    dataset_test, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn
)

gpu_device = torch.device("cuda:0")
object_detector = detector(
    train=False,
    object_classes=dataset_test.object_classes,
    use_SUPPLY=True,
    mode=conf.mode,
).to(device=gpu_device)
object_detector.eval()


model = STTran(
    mode=conf.mode,
    attention_class_num=len(dataset_test.attention_relationships),
    spatial_class_num=len(dataset_test.spatial_relationships),
    contact_class_num=len(dataset_test.contacting_relationships),
    obj_classes=dataset_test.object_classes,
    enc_layer_num=conf.enc_layer,
    dec_layer_num=conf.dec_layer,
).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt["state_dict"], strict=False)
print("*" * 50)
print("CKPT {} is loaded".format(conf.model_path))
"""
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)
"""

if conf.phrdet_mode:
    evaluator3 = BasicSceneGraphEvaluator(
        mode='phrdet',
        AG_object_classes=dataset_test.object_classes,
        AG_all_predicates=dataset_test.relationship_classes,
        AG_attention_predicates=dataset_test.attention_relationships,
        AG_spatial_predicates=dataset_test.spatial_relationships,
        AG_contacting_predicates=dataset_test.contacting_relationships,
        iou_threshold=0.5,
        constraint="no",
    )
else:
    evaluator3 = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=dataset_test.object_classes,
        AG_all_predicates=dataset_test.relationship_classes,
        AG_attention_predicates=dataset_test.attention_relationships,
        AG_spatial_predicates=dataset_test.spatial_relationships,
        AG_contacting_predicates=dataset_test.contacting_relationships,
        iou_threshold=0.5,
        constraint="no",
    )


with torch.no_grad():
    for b, data in enumerate(dataloader):

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = dataset_test.gt_annotations[data[4]]

        entry = object_detector(
            im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None
        )

        pred = model(entry)
        # evaluator1.evaluate_scene_graph(gt_annotation, dict(pred))
        # evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
        evaluator3.evaluate_scene_graph(gt_annotation, dict(pred))

with open("results/results.json", "w") as fp:
    json.dump(evaluator3.results_targets, fp)


#print("-------------------------with constraint-------------------------------")
# evaluator1.print_stats()
#print("-------------------------semi constraint-------------------------------")
# evaluator2.print_stats()
print("-------------------------no constraint-------------------------------")
evaluator3.print_stats()
evaluator3.print_video_stats()
