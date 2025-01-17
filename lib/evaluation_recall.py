import torch
import cv2
import os
import os.path as op
import torch.nn as nn
import numpy as np
from functools import reduce
from lib.ults.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from PIL import Image
import json

class BasicSceneGraphEvaluator:
    def __init__(self, mode, AG_object_classes, AG_all_predicates, AG_attention_predicates, AG_spatial_predicates, AG_contacting_predicates,
                 iou_threshold=0.5, constraint=False, semithreshold=None):
        self.k_list = [1, 3, 5, 10, 20, 50]
        self.result_dict = {}
        self.tot_result_dict = {}
        self.video_result_dict = {}
        self.video_gt_cnt = {}

        self.mode = mode
        #self.result_dict[self.mode + '_recall'] = {1:[], 3:[], 5:[], 10: [], 20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_recall'] = {k: [] for k in self.k_list}

        self.constraint = constraint # semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semithreshold = semithreshold
        self.results_targets = {}

    def reset_result(self):
        #self.result_dict[self.mode + '_recall'] = {1:[], 3:[], 5:[], 10: [], 20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_recall'] = {k: [] for k in self.k_list}

    def print_video_stats(self):
        print('======================' + self.mode + ' video ============================')
        tot_result_dict = {k: [] for k in self.k_list}
        for video_id, v in self.video_result_dict.items():
            for k, vals in v.items():
                self.video_result_dict[video_id][k] = float(vals) / float(self.video_gt_cnt[video_id])
                tot_result_dict[k].append(self.video_result_dict[video_id][k])
        for k, v in tot_result_dict.items():
            print('R@%i: %f' % (k, np.mean(v)))

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))

    def evaluate_scene_graph(self, gt, pred):
        '''collect the groundtruth and prediction'''

        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)

        for idx, frame_gt in enumerate(gt):
            video_id = frame_gt[1]["metadata"]["tag"].split("/")[0]
            if video_id not in self.video_result_dict:
                self.video_result_dict[video_id] = {k: 0 for k in self.k_list}
                self.video_gt_cnt[video_id] = 0

            # generate the ground truth
            gt_boxes = np.zeros([len(frame_gt), 4]) #now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):
                # each pair
                gt_boxes[m+1,:] = n['bbox']
                gt_classes[m+1] = n['class']
                gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])]) # for attention triplet <human-object-predicate>_
                #spatial and contacting relationship could be multiple
                for spatial in n['spatial_relationship'].numpy().tolist():
                    gt_relations.append([m+1, human_idx, self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])]) # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():
                    gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
            }

            # first part for attention and contact, second for spatial

            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),             #attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:,::-1],     #spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()), axis=0)    #contacting


            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_3 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)

            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:,1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }

            self.save_results(pred_entry, gt_entry, frame_gt)

            recalls = evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict, self.k_list,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semithreshold)

            for k in recalls:
                self.result_dict[self.mode + '_recall'][k].append(recalls[k] / float(gt_entry["gt_relations"].shape[0]))
                self.video_result_dict[video_id][k] += recalls[k]

            self.video_gt_cnt[video_id] += gt_entry["gt_relations"].shape[0]


    def save_results(self, pred_entry, gt_entry, frame_gt):
        gt_rels = gt_entry['gt_relations']
        gt_boxes = gt_entry['gt_boxes'].astype(float)
        gt_classes = gt_entry['gt_classes']

        pred_rel_inds = pred_entry['pred_rel_inds']
        rel_scores = pred_entry['rel_scores']


        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']

        method = self.constraint
        if method == 'semi':
            pred_rels = []
            predicate_scores = []
            for i, j in enumerate(pred_rel_inds):
                if rel_scores[i,0]+rel_scores[i,1] > 0:
                    # this is the attention distribution
                    pred_rels.append(np.append(j,rel_scores[i].argmax()))
                    predicate_scores.append(rel_scores[i].max())
                elif rel_scores[i,3]+rel_scores[i,4] > 0:
                    # this is the spatial distribution
                    for k in np.where(rel_scores[i]>threshold)[0]:
                        pred_rels.append(np.append(j, k))
                        predicate_scores.append(rel_scores[i,k])
                elif rel_scores[i,9]+rel_scores[i,10] > 0:
                    # this is the contact distribution
                    for k in np.where(rel_scores[i]>threshold)[0]:
                        pred_rels.append(np.append(j, k))
                        predicate_scores.append(rel_scores[i,k])

            pred_rels = np.array(pred_rels)
            predicate_scores = np.array(predicate_scores)
        elif method == 'no':
            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
            overall_scores = obj_scores_per_rel[:, None] * rel_scores
            score_inds = argsort_desc(overall_scores)[:100]
            pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
            predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
        else:
            pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
            predicate_scores = rel_scores.max(1)

        if pred_rels.size == 0:
            return [[]], np.zeros((0,5)), np.zeros(0)

        frame_id = frame_gt[0]['frame_id']
        self.results_targets[frame_id] = {
            "result": {
                "boxes": pred_boxes.tolist(),
                "labels": pred_classes.tolist(),
                "rels": pred_rels.tolist(),
                "scores": obj_scores.tolist(),
                "rel_scores": predicate_scores.tolist(),
            },
            "target": {
                "rels": gt_rels.tolist(),
                "boxes": gt_boxes.tolist(),
                "labels": gt_classes.tolist(),
            },
        }


    def visualize(self, gt, pred, data_path, text=True, relation=True):
        label_to_idx, idx_to_label = {}, {}
        with open(op.join(data_path, "annotations/object_classes.txt"), "r") as file:
            lines = file.readlines()
            class_id = 1
            for line in lines:
                label_to_idx[line.rstrip()] = class_id
                idx_to_label[class_id] = line.rstrip()
                class_id += 1
        self.idx_to_label = idx_to_label

        predicate_to_idx, idx_to_predicate = {}, {}
        with open(
            op.join(data_path, "annotations/relationship_classes.txt"), "r"
        ) as file:
            lines = file.readlines()
            class_id = 0
            for line in lines:
                predicate_to_idx[line.rstrip()] = class_id
                idx_to_predicate[class_id] = line.rstrip()
                class_id += 1
        self.idx_to_predicate = idx_to_predicate

        pred['attention_distribution'] = nn.functional.softmax(pred['attention_distribution'], dim=1)

        results_targets = {}
        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            gt_boxes = np.zeros([len(frame_gt), 4]) #now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):
                # each pair
                gt_boxes[m+1,:] = n['bbox']
                gt_classes[m+1] = n['class']
                gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_attention_predicates[n['attention_relationship']])]) # for attention triplet <human-object-predicate>_
                #spatial and contacting relationship could be multiple
                for spatial in n['spatial_relationship'].numpy().tolist():
                    gt_relations.append([m+1, human_idx, self.AG_all_predicates.index(self.AG_spatial_predicates[spatial])]) # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():
                    gt_relations.append([human_idx, m+1, self.AG_all_predicates.index(self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>

            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
            }

            # first part for attention and contact, second for spatial

            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),             #attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:,::-1],     #spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()), axis=0)    #contacting


            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_3 = np.concatenate((np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                                            pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)

            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:,1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }

            gt_rels = gt_entry['gt_relations']
            gt_boxes = gt_entry['gt_boxes'].astype(float)
            gt_classes = gt_entry['gt_classes']

            pred_rel_inds = pred_entry['pred_rel_inds']
            rel_scores = pred_entry['rel_scores']


            pred_boxes = pred_entry['pred_boxes'].astype(float)
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']

            method = self.constraint
            if method == 'semi':
                pred_rels = []
                predicate_scores = []
                for i, j in enumerate(pred_rel_inds):
                    if rel_scores[i,0]+rel_scores[i,1] > 0:
                        # this is the attention distribution
                        pred_rels.append(np.append(j,rel_scores[i].argmax()))
                        predicate_scores.append(rel_scores[i].max())
                    elif rel_scores[i,3]+rel_scores[i,4] > 0:
                        # this is the spatial distribution
                        for k in np.where(rel_scores[i]>threshold)[0]:
                            pred_rels.append(np.append(j, k))
                            predicate_scores.append(rel_scores[i,k])
                    elif rel_scores[i,9]+rel_scores[i,10] > 0:
                        # this is the contact distribution
                        for k in np.where(rel_scores[i]>threshold)[0]:
                            pred_rels.append(np.append(j, k))
                            predicate_scores.append(rel_scores[i,k])

                pred_rels = np.array(pred_rels)
                predicate_scores = np.array(predicate_scores)
            elif method == 'no':
                obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
                overall_scores = obj_scores_per_rel[:, None] * rel_scores
                score_inds = argsort_desc(overall_scores)[:100]
                pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
                predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
            else:
                pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
                predicate_scores = rel_scores.max(1)

            if pred_rels.size == 0:
                return [[]], np.zeros((0,5)), np.zeros(0)

            """
            frame_id = frame_gt[0]['frame_id']
            results_targets[frame_id] = {
                "result": {
                    "boxes": pred_boxes.tolist(),
                    "labels": pred_classes.tolist(),
                    "rels": pred_rels.tolist(),
                    "scores": obj_scores.tolist(),
                    "rel_scores": rel_scores.tolist(),
                },
                "target": {
                    "rels": gt_rels.tolist(),
                    "boxes": gt_boxes.tolist(),
                    "labels": gt_classes.tolist(),
                },
            }
            """

            num_gt_boxes = gt_boxes.shape[0]
            num_gt_relations = gt_rels.shape[0]
            assert num_gt_relations != 0

            gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                    gt_rels[:, :2],
                                                    gt_classes,
                                                    gt_boxes)
            num_boxes = pred_boxes.shape[0]
            assert pred_rels[:,:2].max() < pred_classes.shape[0]

            # Exclude self rels
            # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
            #assert np.all(pred_rels[:,2] > 0)

            cls_scores=obj_scores
            pred_triplets, pred_triplet_boxes, relation_scores = \
              _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                       predicate_scores, cls_scores)

            #relation_scores = rel_scores
            sorted_scores = relation_scores.prod(1)
            pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
            pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
            relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
            scores_overall = relation_scores.prod(1)

            top = 5
            frame_name = frame_gt[1]["metadata"]["tag"].split("/")
            frame_name = "/".join(frame_name[:-2] + frame_name[-1:]) + ".png"
            video_id = frame_name.split("/")[0]
            image = Image.open(op.join(data_path, "frames", frame_name))
            triplets, boxes = pred_triplets[:top], pred_triplet_boxes[:top]
            box_img = np.array(image.copy())
            box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR)
            for i, (triplet, box) in enumerate(zip(triplets, boxes)):
                if triplet[1].item() == -1:
                    continue
                sub, obj = (
                    self.idx_to_label[triplet[0].item()],
                    self.idx_to_label[triplet[2].item()],
                )
                sub_box, obj_box = (
                    np.round(box[:4]).astype(int),
                    np.round(box[4:]).astype(int),
                )
                cv2.rectangle(box_img , sub_box[:2], sub_box[2:], OBJECT_TO_BGR[sub], thickness=4)
                cv2.rectangle(box_img , obj_box[:2], obj_box[2:], OBJECT_TO_BGR[obj], thickness=4)

            basename, ext = op.splitext(frame_name)
            output_name = op.join("./results", f"{basename}_box{ext}")
            if not op.exists(op.dirname(output_name)):
                os.mkdir(op.dirname(output_name))
            cv2.imwrite(output_name, box_img)

            for i, (triplet, box) in enumerate(zip(triplets, boxes)):
                img = np.array(image.copy())
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if triplet[1].item() == -1:
                    continue
                sub, pre, obj = (
                    self.idx_to_label[triplet[0].item()],
                    self.idx_to_predicate[triplet[1].item()],
                    self.idx_to_label[triplet[2].item()],
                )
                sub_box, obj_box = (
                    np.round(box[:4]).astype(int),
                    np.round(box[4:]).astype(int),
                )
                cv2.rectangle(img, sub_box[:2], sub_box[2:], OBJECT_TO_BGR[sub], thickness=2)
                cv2.rectangle(
                    img, obj_box[:2], obj_box[2:], OBJECT_TO_BGR[obj], thickness=2
                )
                if text:
                    cv2.putText(
                        img,
                        text=sub,
                        org=sub_box[:2] - np.array([0, 5]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=OBJECT_TO_BGR[sub],
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        text=obj,
                        org=obj_box[:2] - np.array([0, 5]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.7,
                        color=OBJECT_TO_BGR[obj],
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                sub_center, obj_center = (
                    (sub_box[:2] + sub_box[2:]) // 2,
                    (obj_box[:2] + obj_box[2:]) // 2,
                )
                if relation:
                    cv2.line(
                        img,
                        sub_center,
                        obj_center,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_4,
                        shift=0,
                    )
                    if text:
                        cv2.putText(
                            img,
                            text=pre,
                            org=(sub_center + obj_center) // 2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7,
                            color=(0, 255, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

                basename, ext = op.splitext(frame_name)
                output_name = op.join("./results", f"{basename}_rel_top{i}{ext}")
                if not op.exists(op.dirname(output_name)):
                    os.mkdir(op.dirname(output_name))
                cv2.imwrite(output_name, img)

        """
        with open("results/results.json", "w") as fp:
            json.dump(results_targets, fp)
        """



def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, k_list, method=None, threshold = 0.9, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']


    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']

    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i,0]+rel_scores[i,1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j,rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i,3]+rel_scores[i,4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])
            elif rel_scores[i,9]+rel_scores[i,10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i]>threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i,k])

        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]

    else:
        pred_rels = np.column_stack((pred_rel_inds, rel_scores.argmax(1))) #1+  dont add 1 because no dummy 'no relations'
        predicate_scores = rel_scores.max(1)


    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, pred_boxes, pred_classes,
                predicate_scores, obj_scores, phrdet= mode=='phrdet',
                **kwargs)

    recalls = {k: 0 for k in k_list}
    for k in result_dict[mode + '_recall']:
        matchs = reduce(np.union1d, pred_to_gt[:k])
        #rec_i = float(len(matchs)) / float(gt_rels.shape[0])
        #result_dict[mode + '_recall'][k].append(rec_i)
        recalls[k] = float(len(matchs))

    return recalls
    #return pred_to_gt, pred_5ples, rel_scores

###########################
def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, pred_boxes, pred_classes, rel_scores=None, cls_scores=None,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:,:2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    #assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:,2], pred_rels[:,:2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)

    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:,:2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

OBJECT_TO_BGR = {
    "person": (0, 0, 255),
    "bag": (13, 76, 231),
    "bed": (252, 8, 102),
    "blanket": (255, 13, 145),
    "book": (8, 90, 192),
    "box": (154, 62, 6),
    "broom": (6, 184, 163),
    "chair": (132, 203, 16),
    "closetcabinet": (4, 152, 7),
    "clothes": (4, 152, 7),
    "cupglassbottle": (132, 214, 220),
    "dish": (157, 111, 190),
    "door": (152, 227, 68),
    "doorknob": (214, 193, 41),
    "doorway": (217, 69, 116),
    "floor": (14, 117, 4),
    "food": (243, 20, 181),
    "groceries": (203, 171, 24),
    "laptop": (173, 135, 122),
    "light": (88, 39, 124),
    "medicine": (66, 64, 83),
    "mirror": (82, 132, 38),
    "papernotebook": (129, 231, 49),
    "phonecamera": (157, 74, 152),
    "picture": (138, 69, 109),
    "pillow": (124, 13, 250),
    "refrigerator": (63, 62, 109),
    "sandwich": (100, 142, 74),
    "shelf": (128, 246, 185),
    "shoe": (167, 112, 96),
    "sofacouch": (98, 156, 162),
    "table": (95, 170, 5),
    "television": (117, 174, 248),
    "towel": (144, 179, 189),
    "vacuum": (126, 106, 135),
    "window": (201, 178, 116),
    "turtle": (227, 56, 70),
    "antelope": (108, 232, 158),
    "bicycle": (206, 239, 174),
    "lion": (34, 116, 19),
    "ball": (118, 69, 123),
    "motorcycle": (148, 34, 227),
    "cattle": (198, 23, 12),
    "airplane": (190, 137, 27),
    "red_panda": (112, 200, 140),
    "horse": (187, 188, 102),
    "watercraft": (18, 94, 106),
    "monkey": (7, 255, 141),
    "fox": (37, 192, 160),
    "elephant": (36, 28, 186),
    "bird": (169, 218, 231),
    "sheep": (106, 135, 153),
    "frisbee": (221, 125, 65),
    "giant_panda": (120, 122, 105),
    "squirrel": (141, 236, 46),
    "bus": (70, 47, 189),
    "bear": (220, 185, 227),
    "tiger": (13, 202, 247),
    "train": (215, 20, 57),
    "snake": (12, 134, 220),
    "rabbit": (38, 100, 248),
    "whale": (27, 86, 172),
    "sofa": (248, 65, 255),                                                                                                              "skateboard": (84, 205, 165),
    "dog": (178, 186, 40),
    "domestic_cat": (10, 48, 10),
    "lizard": (21, 130, 135),
    "hamster": (160, 45, 203),
    "car": (86, 132, 169),
    "zebra": (0, 213, 136),
}
