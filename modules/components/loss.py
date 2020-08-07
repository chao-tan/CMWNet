import torch
from torch.nn import functional as F


def EPE(flow_predict, flow_target):
    batch_size, _, h, w = flow_predict.shape
    flow_target = F.interpolate(flow_target, (h, w), mode='area')
    return torch.norm(flow_predict - flow_target, 2, 1).mean()


def MultiEPE(flows_predicts, flow_target, weights=(0.005, 0.01, 0.02, 0.08, 0.32)):
    if len(flows_predicts) < 5:
        weights = [0.005]*len(flows_predicts)

    loss = 0
    for i in range(len(weights)):
        loss += weights[i] * EPE(flows_predicts[i], flow_target)
    return loss