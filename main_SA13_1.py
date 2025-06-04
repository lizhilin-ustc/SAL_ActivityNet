import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import core.utils as utils
from core.SA13_2 import SALModel
from core.config13_base import build_args
from core.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from eval.eval_detection13 import ANETdetection, nms
from terminaltables import AsciiTable
from core.dataset_class13 import build_dataset
from torch.utils.data import DataLoader
from eval1 import misc_utils, eval_detection
from core.loss_asl import CrossEntropyLoss, GeneralizedCE, GeneralizedCE_Mask
import torch.nn.functional as F
import torch.nn as nn

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        self.ulb_prob_t = torch.ones((cfg.NUM_CLASSES)).cuda() / cfg.NUM_CLASSES
        self.prob_max_mu_t = 1.0 / cfg.NUM_CLASSES
        self.prob_max_var_t = 1.0
        self.ema_p = 0.99  

    @torch.no_grad()
    def update_prob_t(self, ulb_probs):
        # import pdb;pdb.set_trace()
        ulb_prob_t = ulb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1 - self.ema_p) * ulb_prob_t

        max_probs, max_idx = ulb_probs.max(dim=-1)
        prob_max_mu_t = torch.mean(max_probs)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1 - self.ema_p) * prob_max_mu_t.item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1 - self.ema_p) * prob_max_var_t.item()
        # print(self.prob_max_mu_t, self.prob_max_var_t)

    @torch.no_grad()
    def calculate_mask(self, probs):
        max_probs, max_idx = probs.max(dim=-1)

        # compute weight
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t
        # mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (4 * var)))   # v=4
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (6 * var)))
        return max_probs.detach(), mask.detach()
    
    def train_one_step(self, net, loader_iter, optimizer, writter, step):
        net.train()

        data, label= next(loader_iter)
        # vid_name, input_feature, vid_label_t, vid_len, vid_duration
        data = data.cuda()
        label = label.cuda()
        batch_size = data.shape[0]

        optimizer.zero_grad()

        cas, cas_bg, action_flow, action_rgb, action_flow_aug, action_rgb_aug, sc, sc_flow, sc_rgb = net(data)
        num_segments = cas.shape[1]
        # import pdb; pdb.set_trace()

        _, topk_indices = torch.topk(cas, num_segments // 4, dim=1)  #
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        action = 0.7 * action_flow.permute(0, 2, 1) + 0.3 * action_rgb.permute(0, 2, 1)
        _, topk_indices_action = torch.topk(action, 40, dim=1)

        background = 1 - action    # [16, 750, 1]

        _, topk_indices_bg = torch.topk(torch.softmax(cas_bg,-1), num_segments // 8, dim=1)    ### K=?
        cas_top_bg = torch.mean(torch.gather(cas_bg, 1, topk_indices_bg), dim=1)   

        cls_agnostic_gt = calculate_pesudo_target1(batch_size, topk_indices_action, num_segments)
        cls_agnostic_gt_aug = (cls_agnostic_gt + torch.cat((cls_agnostic_gt[-1:], cls_agnostic_gt[:-1]),dim=0))
        cls_agnostic_gt_aug = torch.where(cls_agnostic_gt_aug>=1, 1, 0)

        self.update_prob_t(torch.softmax(cas,dim=-1).reshape(-1, cas.shape[-1]))
        max_probs, mask = self.calculate_mask(torch.softmax(cas,dim=-1).reshape(-1, cas.shape[-1]))
        mask = mask.reshape(cas.shape[0], -1)
        mask_aug = (mask + torch.cat((mask[-1:], mask[:-1]),dim=0))/2


        criterion = CrossEntropyLoss()
        gce = GeneralizedCE(q=self.cfg.q)
        gce_mask = GeneralizedCE_Mask(q=self.cfg.q)

        base_loss = criterion(cas_top, label)
        sta_loss = gce_mask(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1), mask) + gce_mask(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1), mask)
        aug_loss = gce_mask(action_flow_aug.squeeze(1), cls_agnostic_gt_aug.squeeze(1), mask_aug) + gce_mask(action_rgb_aug.squeeze(1), cls_agnostic_gt_aug.squeeze(1), mask_aug)
        mu_loss = F.mse_loss(action_rgb, action_flow)
        sc_loss = criterion(sc, label) + criterion(sc_flow, label) + criterion(sc_rgb, label) 

        cas_top_bg = torch.softmax(cas_top_bg, -1)
        label_bg = torch.ones_like(label).cuda()
        label_bg = torch.softmax(label_bg, -1)
        fp_loss = F.kl_div(cas_top_bg.log(), label_bg, reduction='batchmean')

        cost = base_loss + sta_loss + 20*mu_loss + fp_loss + 3*sc_loss + aug_loss 
        loss = {}

        cost.backward()
        optimizer.step()

        for key in loss.keys():
            writter.add_scalar(key, loss[key].cpu().item(), step)
        return cost

    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg.GPU_ID
        worker_init_fn = None
        if self.cfg.SEED >= 0:
            utils.set_seed(self.cfg.SEED)
            worker_init_fn = np.random.seed(self.cfg.SEED)

        utils.set_path(self.cfg)
        utils.save_config(self.cfg)

        net = ASLModel(self.cfg.FEATS_DIM, self.cfg.NUM_CLASSES, self.cfg)
        net = net.cuda()

        train_dataset = build_dataset(self.cfg, phase="train", sample="random")
        test_dataset = build_dataset(self.cfg, phase="test", sample="uniform")

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                        num_workers=self.cfg.num_workers, drop_last=True)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                        num_workers=self.cfg.num_workers, drop_last=False)

        test_info = {"step": [], "test_acc": [], "average_mAP": [],
                    "mAP@0.50": [], "mAP@0.55": [], "mAP@0.60": [],
                    "mAP@0.65": [], "mAP@0.70": [], "mAP@0.75": [],
                    "mAP@0.80": [], "mAP@0.85": [], "mAP@0.90": [], "mAP@0.95": []}

        best_mAP = -1

        self.cfg.LR = eval(self.cfg.LR)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.cfg.LR[0], betas=(0.9, 0.999), weight_decay=0.0005)

        if self.cfg.MODE == 'test':
            _, _ = self.test_all(net, self.cfg, test_loader, test_info, 0, None, self.cfg.MODEL_FILE)
            utils.save_best_record_thumos(test_info, os.path.join(self.cfg.OUTPUT_PATH, "best_results.txt"))
            print(utils.table_format(test_info, self.cfg.TIOU_THRESH, '[SA] ACT1.3 Performance'))
            return
        else:
            writter = SummaryWriter(self.cfg.LOG_PATH)
            
        print('=> test frequency: {} steps'.format(self.cfg.TEST_FREQ))
        print('=> start training...')
        for step in range(1, self.cfg.NUM_ITERS + 1):
            if step > 1 and self.cfg.LR[step - 1] != self.cfg.LR[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.cfg.LR[step - 1]

            if (step - 1) % len(train_loader) == 0:
                loader_iter = iter(train_loader)

            batch_time = AverageMeter()
            losses = AverageMeter()
            
            end = time.time()
            cost = self.train_one_step(net, loader_iter, optimizer, writter, step)
            losses.update(cost.item(), self.cfg.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            if step == 1 or step % self.cfg.PRINT_FREQ == 0:
                print(('Step: [{0:04d}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        step, self.cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))
                
            if step > -1 and step % self.cfg.TEST_FREQ == 0:
                mAP_50, mAP_AVG = self.test_all(net, self.cfg, test_loader, test_info, step, writter)

                if test_info["average_mAP"][-1] > best_mAP:
                    best_mAP = test_info["average_mAP"][-1]
                    best_test_info = copy.deepcopy(test_info)

                    utils.save_best_record_thumos(test_info, os.path.join(self.cfg.OUTPUT_PATH, "best_results.txt"))

                    torch.save(net.state_dict(), os.path.join(self.cfg.MODEL_PATH, "model_best.pth.tar"))

                print(('- Test result: \t' \
                        'mAP@0.5 {mAP_50:.2%}\t' \
                        'mAP@AVG {mAP_AVG:.2%} (best: {best_mAP:.2%})'.format(
                        mAP_50=mAP_50, mAP_AVG=mAP_AVG, best_mAP=best_mAP)))

        print(utils.table_format(best_test_info, self.cfg.TIOU_THRESH, '[SA] ACT1.3 Performance'))

    @torch.no_grad()
    def test_all(self, net, cfg, test_loader, test_info, step, writter=None, model_file=None):
        net.eval()

        if model_file:
            print('=> loading model: {}'.format(model_file))
            net.load_state_dict(torch.load(model_file))
            print('=> tesing model...')

        final_res = {'method': '[SAL]', 'results': {}}
        
        acc = AverageMeter()

        for vid, data, label, vid_num_seg, vid_duration in test_loader:
            data, label = data.cuda(), label.cuda()

            cas, cas_bg, action_flow, action_rgb, action_flow_aug, action_rgb_aug, sc, sc_flow, sc_rgb = net(data)
            num_segments = cas.shape[1]

            combined_cas = 0.6*torch.softmax(cas, -1) + 0.2*action_rgb.permute(0, 2, 1) + 0.2*action_flow.permute(0, 2, 1)
            _, topk_indices = torch.topk(combined_cas, num_segments // 4, dim=1)  #

            cas_top = torch.gather(cas, 1, topk_indices)
            cas_top = torch.mean(cas_top, dim=1)

            beta=0.5   # 0.5
            cas_top = beta*cas_top + (1-beta)*sc  # 

            score_supp = F.softmax(cas_top, dim=1)
            label_np = label.cpu().data.numpy()
            score_np = score_supp[0, :].cpu().data.numpy()

            score_np[np.where(score_np < cfg.CLASS_THRESH)] = 0
            score_np[np.where(score_np >= cfg.CLASS_THRESH)] = 1

            if np.all(score_np == 0):
                arg = np.argmax(score_supp[0, :].cpu().data.numpy())
                score_np[arg] = 1

            correct_pred = np.sum(label_np == score_np, axis=1)

            pred = np.where(score_np > cfg.CLASS_THRESH)[0]

            # action prediction
            if len(pred) != 0:
                cas_pred = combined_cas[0].cpu().numpy()[:, pred]
                cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
                cas_pred = misc_utils.upgrade_resolution(cas_pred, cfg.UP_SCALE)

                proposal_dict = {}

                for t in range(len(cfg.CAS_THRESH)):
                    cas_temp = cas_pred.copy()
                    zero_location = np.where(cas_temp[:, :, 0] < cfg.CAS_THRESH[t])
                    cas_temp[zero_location] = 0

                    cas_input = cas_pred.copy()

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = misc_utils.get_proposal_oic(seg_list, cas_pred.copy(), score_supp[0, :].cpu().data.numpy(),
                                                            pred, cfg.UP_SCALE, vid_num_seg[0].cpu().item(),
                                                            cfg.FEATS_FPS, num_segments, cfg.gamma)

                    for j in range(len(proposals)):
                        if not proposals[j]:
                            continue
                        class_id = proposals[j][0][0]

                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []

                        proposal_dict[class_id] += proposals[j]

                final_proposals = []
                for class_id in proposal_dict.keys():
                    final_proposals.append(misc_utils.basnet_nms(proposal_dict[class_id], cfg.NMS_THRESH,
                                                                cfg.SOFT_NMS, cfg.NMS_ALPHA))

                # print(final_proposals)
                # print(np.array(final_proposals).shape[1])
                # exit()

                final_res['results'][vid[0]] = utils.result2json13(final_proposals, cfg.class_name_lst)

        json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
        json.dump(final_res, open(json_path, 'w'))
        
        anet_detection = ANETdetection(cfg.GT_PATH, json_path,subset='val', tiou_thresholds=cfg.TIOU_THRESH,
                                    verbose=False, check_status=False)
        mAP, average_mAP = anet_detection.evaluate()

        if writter:
            writter.add_scalar('Test Performance/Accuracy', acc.avg, step)
            writter.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
            for i in range(cfg.TIOU_THRESH.shape[0]):
                writter.add_scalar('mAP@tIOU/mAP@{:.2f}'.format(cfg.TIOU_THRESH[i]), mAP[i], step)

        test_info["step"].append(step)
        test_info["test_acc"].append(acc.avg)
        test_info["average_mAP"].append(average_mAP)

        for i in range(cfg.TIOU_THRESH.shape[0]):
            test_info["mAP@{:.2f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
        return test_info['mAP@0.50'][-1], average_mAP


def calculate_pesudo_target1(batch_size, topk_indices, num_segments):
    cls_agnostic_gt = []
    cls_agnostic_neg_gt = []
    for b in range(batch_size):
        topk_indices_b = topk_indices[b, :]          # topk, num_actions
        cls_agnostic_gt_b = torch.zeros((1, 1, num_segments)).cuda()

        cls_agnostic_gt_b[0, 0, topk_indices_b[:, 0]] = 1
        cls_agnostic_gt.append(cls_agnostic_gt_b)

    return torch.cat(cls_agnostic_gt, dim=0)         # B, 1, num_segments




def main():
    cfg = build_args(dataset="ActivityNet")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()