import os
import os.path as op
import json

files = os.listdir("train")

sbj_counts = 0
for f in files:
    data = json.load(open(op.join("train", f)))
    relationships = [[] for _ in range(len(data["trajectories"]))]
    for rel in data["relation_instances"]:
        for fid in range(rel["begin_fid"], rel["end_fid"]):
            relationships[fid].append(
                {
                    "sub": rel["subject_tid"],
                    "obj": rel["object_tid"],
                }
            )

    frame_mean_subjects = 0
    for fid in range(len(data["trajectories"])):
        subject_tids = {}
        rels = relationships[fid]
        for rel in rels:
            subject_tids[rel["sub"]] = 1
        frame_mean_subjects += len(subject_tids)

    print("Frame mean", frame_mean_subjects / len(data["trajectories"]))

    sbj_counts += frame_mean_subjects / len(data["trajectories"])


print("Sbj counts", sbj_counts / len(files))
