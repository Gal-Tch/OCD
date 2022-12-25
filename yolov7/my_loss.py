from data_loader import wrapper_dataset
from utils.metrics import ap_per_class
import numpy as np
import torch
import copy
from utils.general import box_iou, non_max_suppression, xywh2xyxy,
from utils_OCD import recursivley_detach


def yolo_loss(out, targets):
    conf_thres = 0.001
    iou_thres = 0.6
    nc = 80
    device = "cuda"
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    #  todo: in origina batch loop

    targets = targets.to(device)

    # Run NMS
    # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # todo: to pixels
    lb = []  # for autolabelling
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

    # Statistics per image
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map), maps


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    args = dotdict()
    args["datatype"] = "yolov7-tiny.py"
    train_loader, test_loader, model = wrapper_dataset("", args, "cuda")
    batch = next(test_loader)
    predicted_labels, h = model(batch['input'].float())
    hx, hy = h
    hfirst = copy.deepcopy((hx.detach(), hy.detach()))
    out = copy.deepcopy(recursivley_detach(predicted_labels))
    loss = yolo_loss(predicted_labels, batch['output'].long())
