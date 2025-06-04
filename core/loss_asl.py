import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        return first_term + second_term

class BCE(nn.Module):
    def __init__(self, ):
        super(BCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        first_term = - torch.mean(torch.sum((logits + 1e-7).log() * label, dim=1) / pos_factor)
        second_term = - torch.mean(torch.sum((1 - logits + 1e-7).log() * (1 - label), dim=1) / neg_factor)

        return first_term + second_term   

class TotalLoss(nn.Module):
    def __init__(self, q):
        super(TotalLoss, self).__init__()
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=q)


    def forward(self, cas_top, _label, action_flow, action_rgb, cls_agnostic_gt):
        base_loss = self.criterion(cas_top, _label)
        cost = base_loss

        cls_agnostic_loss_flow = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1))
        cls_agnostic_loss_rgb = self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        cost += cls_agnostic_loss_flow + cls_agnostic_loss_rgb

        loss_dict = {
            'Loss/base': base_loss,
            'Loss/flow': cls_agnostic_loss_flow,
            'Loss/rgb': cls_agnostic_loss_rgb
        }

        return cost, loss_dict


class GeneralizedCE_Mask(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE_Mask, self).__init__()

    def forward(self, logits, label, mask):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label * mask, dim=1) + 1e-7
        neg_factor = torch.sum((1 - label) * mask, dim=1) + 1e-7

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label * mask, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label) * mask, dim=1)/neg_factor)

        return first_term + second_term