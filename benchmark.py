import numpy as np
import time

np.set_printoptions(precision=4)
import copy

import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader.action_genome import AG, cuda_collate_fn
from lib.config import Config
from lib.object_detector import detector
from lib.sttran import STTran

from lib.ults.flop_count import flop_count, fmt_res

conf = Config()
for i in conf.args:
    print(i, ":", conf.args[i])
AG_dataset = AG(
    mode="test",
    datasize=conf.datasize,
    data_path=conf.data_path,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == "predcls" else True,
)
dataloader = torch.utils.data.DataLoader(
    AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn
)

gpu_device = torch.device("cuda:0")
object_detector = detector(
    train=False,
    object_classes=AG_dataset.object_classes,
    use_SUPPLY=True,
    mode=conf.mode,
).to(device=gpu_device)
object_detector.eval()

model = STTran(
    mode=conf.mode,
    attention_class_num=len(AG_dataset.attention_relationships),
    spatial_class_num=len(AG_dataset.spatial_relationships),
    contact_class_num=len(AG_dataset.contacting_relationships),
    obj_classes=AG_dataset.object_classes,
    enc_layer_num=conf.enc_layer,
    dec_layer_num=conf.dec_layer,
).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt["state_dict"], strict=False)
print("*" * 50)
print("CKPT {} is loaded".format(conf.model_path))


class CombinedModel(nn.Module):
    def __init__(self, detector, classifier, gt_annotation):
        super().__init__()
        self.detector = detector
        self.classifier = classifier
        self.gt_annotation = gt_annotation

    def forward(self, entry):
        entry = self.detector(entry[0], entry[1], entry[2], entry[3], self.gt_annotation, im_all=None)
        entry = self.classifier(entry)
        return list(entry.values())

speed_time = 0.0
speed_time_including_detector = 0.0
n_iters = 0
tmp = []
with torch.no_grad():
    for b, data in enumerate(tqdm(dataloader)):
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]

        start = time.time()
        entry = object_detector(
            im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None
        )
        elapsed_time = time.time() - start
        speed_time_including_detector += elapsed_time

        start = time.time()
        pred = model(entry)
        elapsed_time = time.time() - start

        combined_model = CombinedModel(object_detector, model, gt_annotation)
        inputs = [im_data, im_info, gt_boxes, num_boxes]
        res = flop_count(combined_model, (inputs,))
        tmp.append(sum(res.values()))

        speed_time += elapsed_time
        speed_time_including_detector += elapsed_time
        n_iters += 1

        if n_iters >= 10:
            break

speed_time /= n_iters
speed_time_including_detector /= n_iters

print("--- Results ---")
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_parameters_inc_detector = n_parameters + sum(p.numel() for p in object_detector.parameters() if p.requires_grad)
print("Number of params:", n_parameters)
print("Number of paramas including detector:", n_parameters_inc_detector)
print("flops", fmt_res(np.array(tmp)))
print("Speed (s/image): ", speed_time)
print("Speed including detector (s/image): ", speed_time_including_detector)
