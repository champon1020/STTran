import argparse
import pickle
import os
import os.path as op
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--vidvrd_path", type=str, default="./data/vidvrd")

frame_list = []
person_bbox = {}
object_bbox_and_relationship = {}

def reorder_relationship_classes():
    relationship_classes = {"a": [], "s": [], "c": []}
    for r, t in relationship_types.items():
        relationship_classes[t].append(r)
    print("a:", len(relationship_classes["a"]))
    print("s:", len(relationship_classes["s"]))
    print("c:", len(relationship_classes["c"]))
    return relationship_classes["a"] + relationship_classes["s"] + relationship_classes["c"]

def convert_rel_type(rels):
    s, a, c = [], [], []
    for rel in rels:
        if relationship_types[rel] == "s":
            s.append(rel)
        elif relationship_types[rel] == "a":
            a.append(rel)
        elif relationship_types[rel] == "c":
            c.append(rel)
    return s, a, c

relationship_types = {
    "taller": "s",
    "swim_behind": "s",
    "walk_away": "a",
    "fly_behind": "s",
    "creep_behind": "s",
    "lie_with": "a",
    "move_left": "s",
    "stand_next_to": "s",
    "touch": "c",
    "follow": "a",
    "move_away": "a",
    "lie_next_to": "s",
    "walk_with": "c",
    "move_next_to": "s",
    "creep_above": "s",
    "stand_above": "s",
    "fall_off": "a",
    "run_with": "c",
    "swim_front": "s",
    "walk_next_to": "s",
    "kick": "c",
    "stand_left": "s",
    "creep_right": "s",
    "sit_above": "s",
    "watch": "a",
    "swim_with": "c",
    "fly_away": "a",
    "creep_beneath": "s",
    "front": "s",
    "run_past": "s",
    "jump_right": "s",
    "fly_toward": "s",
    "stop_beneath": "s",
    "stand_inside": "s",
    "creep_left": "s",
    "run_next_to": "s",
    "beneath": "s",
    "stop_left": "s",
    "right": "s",
    "jump_front": "s",
    "jump_beneath": "s",
    "past": "s",
    "jump_toward": "s",
    "sit_front": "s",
    "sit_inside": "s",
    "walk_beneath": "s",
    "run_away": "a",
    "stop_right": "s",
    "run_above": "s",
    "walk_right": "s",
    "away": "a",
    "move_right": "s",
    "fly_right": "s",
    "behind": "s",
    "sit_right": "s",
    "above": "s",
    "run_front": "s",
    "run_toward": "s",
    "jump_past": "s",
    "stand_with": "c",
    "sit_left": "s",
    "jump_above": "s",
    "move_with": "c",
    "swim_beneath": "s",
    "stand_behind": "s",
    "larger": "a",
    "walk_past": "s",
    "stop_front": "s",
    "run_right": "s",
    "creep_away": "a",
    "move_toward": "s",
    "feed": "c",
    "run_left": "s",
    "lie_beneath": "s",
    "fly_front": "s",
    "walk_behind": "s",
    "stand_beneath": "s",
    "fly_above": "s",
    "bite": "c",
    "fly_next_to": "s",
    "stop_next_to": "s",
    "fight": "c",
    "walk_above": "s",
    "jump_behind": "s",
    "fly_with": "c",
    "sit_beneath": "s",
    "sit_next_to": "s",
    "jump_next_to": "s",
    "run_behind": "s",
    "move_behind": "s",
    "swim_right": "s",
    "swim_next_to": "s",
    "hold": "c",
    "move_past": "s",
    "pull": "c",
    "stand_front": "s",
    "walk_left": "s",
    "lie_above": "s",
    "ride": "c",
    "next_to": "s",
    "move_beneath": "s",
    "lie_behind": "s",
    "toward": "s",
    "jump_left": "s",
    "stop_above": "s",
    "creep_toward": "s",
    "lie_left": "s",
    "fly_left": "s",
    "stop_with": "c",
    "walk_toward": "s",
    "stand_right": "s",
    "chase": "c",
    "creep_next_to": "s",
    "fly_past": "s",
    "move_front": "s",
    "run_beneath": "s",
    "creep_front": "s",
    "creep_past": "s",
    "play": "a",
    "lie_inside": "s",
    "stop_behind": "s",
    "move_above": "s",
    "sit_behind": "s",
    "faster": "a",
    "lie_right": "s",
    "walk_front": "s",
    "drive": "a",
    "swim_left": "s",
    "jump_away": "a",
    "jump_with": "c",
    "lie_front": "s",
    "left": "s",
    "unsure": "a",
}


def main(args):
    train_files = os.listdir(op.join(args.vidvrd_path, "train"))
    test_files = os.listdir(op.join(args.vidvrd_path, "test"))
    process(train_files, "train")
    process(test_files, "test")
    with open(op.join(args.vidvrd_path, "annotations/person_bbox.pkl"), "wb") as fp:
        pickle.dump(person_bbox, fp)

    with open(op.join(args.vidvrd_path, "annotations/object_bbox_and_relationship.pkl"), "wb") as fp:
        pickle.dump(object_bbox_and_relationship, fp)

    with open(op.join(args.vidvrd_path, "annotations/frame_list.txt"), "r+") as fp:
        fp.writelines("\n".join(frame_list))

    relationship_classes = reorder_relationship_classes()
    with open(op.join(args.vidvrd_path, "annotations/relationship_classes.txt"), "r+") as fp:
        fp.writelines("\n".join(relationship_classes))

def process(annot_files, image_set):
    for annot in tqdm(annot_files):
        data = json.load(open(op.join(args.vidvrd_path, image_set, annot)))
        video_id = data["video_id"]
        w, h = data["width"], data["height"]

        relationships = [{} for _ in range(len(data["trajectories"]))]
        for rel in data["relation_instances"]:
            for fid in range(rel["begin_fid"], rel["end_fid"]):
                if rel["subject_tid"] not in relationships[fid]:
                    relationships[fid][rel["subject_tid"]] = [
                        {
                            "obj": rel["object_tid"],
                            "rel": rel["predicate"],
                        }
                    ]
                else:
                    relationships[fid][rel["subject_tid"]].append(
                        {
                            "obj": rel["object_tid"],
                            "rel": rel["predicate"],
                        }
                    )

        obj_tid_to_label = {o["tid"]: o["category"] for o in data["subject/objects"]}
        for fid, frame in enumerate(data["trajectories"]):
            if len(frame) == 0:
                continue

            image_id = f"{video_id}.mp4/{fid:06d}.jpg"
            frame_list.append(image_id)

            obj_tid_to_bbox = {}
            for ind, bbox in enumerate(frame):
                box = (bbox["bbox"]["xmin"], bbox["bbox"]["ymin"], bbox["bbox"]["xmax"], bbox["bbox"]["ymax"])
                obj_tid_to_bbox[bbox["tid"]] = box

            rels = relationships[fid]
            for sbj_tid, objects in rels.items():
                image_id_with_sbj_tid = image_id + "@" + str(sbj_tid)
                sbj_label = obj_tid_to_label[sbj_tid]
                sbj_bbox =  obj_tid_to_bbox[sbj_tid]
                person_bbox[image_id_with_sbj_tid] = {
                    "bbox": np.array([sbj_bbox]),
                    "bbox_size": (w, h),
                    "bbox_mode": "xyxy",
                }

                objects_dict = {}
                for obj in objects:
                    if obj["obj"] not in objects_dict:
                        objects_dict[obj["obj"]] = [obj["rel"]]
                    else:
                        objects_dict[obj["obj"]].append(obj["rel"])

                object_bbox_and_relationship[image_id_with_sbj_tid] = []
                for obj_tid, rel_labels in objects_dict.items():
                    s, a, c = convert_rel_type(rel_labels)
                    obj_bbox = obj_tid_to_bbox[obj_tid]
                    obj_bbox = (obj_bbox[0], obj_bbox[1], obj_bbox[2]-obj_bbox[0], obj_bbox[3]-obj_bbox[1]) #xywh
                    a_rel = ['unsure'] if len(a) == 0 else [a[0]]
                    obj_dict = {
                        "class": obj_tid_to_label[obj_tid],
                        "bbox": obj_bbox,
                        "attention_relationship": a_rel,
                        "spatial_relationship": s,
                        "contacting_relationship": c,
                        "visible": False if len(a+s+c) == 0 else True,
                        "metadata": {
                            "tag": image_id,
                            "set": image_set,
                        }
                    }
                    object_bbox_and_relationship[image_id_with_sbj_tid].append(obj_dict)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
