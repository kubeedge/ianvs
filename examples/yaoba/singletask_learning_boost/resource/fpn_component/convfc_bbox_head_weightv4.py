import numpy as np
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmcv.runner import force_fp32
import torch
from mmdet.models.losses import accuracy
from mmdet.core import multi_apply


@HEADS.register_module()
class Shared2FCBBoxHeadWeightV4(Shared2FCBBoxHead):

    def __init__(self, **kwargs):
        super(Shared2FCBBoxHeadWeightV4, self).__init__(**kwargs)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             custom_weight,
             gt_labels,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_gt_inds,
             reduction_override=None):
        torch.set_printoptions(threshold=np.inf)
        # 获取有效预测结果的mask,非有效预测的结果弄成零方便取权重,最后通过mask筛选取出的值
        bbox_gt_inds_mask = bbox_gt_inds != -1
        bbox_gt_inds[bbox_gt_inds == -1] = 0
        # 通过bbox的weight和每个bbox属于的类别计算出类别的权重，
        # [1,0.5,0.8] 对应的类别为[0,0,1] 那么类别权重和背景权重为[0.75,0.8,0.76]
        custom_label_weight = []
        for i in range(len(custom_weight)):
            custom_label_weight.append([0 for _ in range(self.num_classes + 1)])
        for i in range(len(custom_label_weight)):
            for j in range(self.num_classes + 1):
                # num_classes代表背景
                if j == self.num_classes:
                    mask = np.asarray(custom_label_weight[i]) > 0
                    background_weight = np.average(np.asarray(custom_label_weight[i])[mask])
                    custom_label_weight[i][j] = background_weight
                else:
                    img_i_gt_labels_wrt_class_j = (gt_labels[i] == j).cpu().numpy()
                    img_i_class_j_weight = custom_weight[i][img_i_gt_labels_wrt_class_j]
                    if len(img_i_class_j_weight) > 0:
                        custom_label_weight[i][j] = np.average(img_i_class_j_weight)
                    else:
                        custom_label_weight[i][j] = 0
        start_index = 0
        lengths = []
        bbox_weight_list = []
        label_weight_list=[]
        predict_img_index = rois[:, 0]
        num_imgs = len(custom_weight)
        # 得出每个img有多少个预测结果,一个img一个img的处理
        for i in range(num_imgs):
            lengths.append(torch.count_nonzero(predict_img_index == i).item())
        for index, length in enumerate(lengths):
            cur_custom_bbox_weight = torch.from_numpy(custom_weight[index]).type_as(bbox_pred)
            cur_custom_label_weight = torch.from_numpy(np.asarray(custom_label_weight[index])).type_as(labels)
            cur_custom_bbox_weight = cur_custom_bbox_weight[bbox_gt_inds[start_index:length + start_index]]
            cur_custom_label_weight = cur_custom_label_weight[labels[start_index:length + start_index]]
            cur_custom_bbox_weight[~bbox_gt_inds_mask[start_index:length + start_index]] = 0
            bbox_weight_list.append(cur_custom_bbox_weight)
            label_weight_list.append(cur_custom_label_weight)
            start_index += length
        final_custom_bbox_weight = torch.concatenate(bbox_weight_list, dim=0)
        final_custom_label_weight = torch.concatenate(label_weight_list, dim=0)
        bbox_weights = final_custom_bbox_weight.unsqueeze(-1) * bbox_weights
        label_weights = final_custom_label_weight * label_weights
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        # 重写这个方法是为了加入pos_assigned_gt_inds,方便判断pos的pred_box是预测的哪个gt_box，在tradboost中每个标签框的权重不一样
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights, bbox_gt_inds = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_assigned_gt_inds_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_gt_inds = torch.cat(bbox_gt_inds, 0)
        return labels, label_weights, bbox_targets, bbox_weights, bbox_gt_inds

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_assigned_gt_inds_list, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        bbox_gt_inds = pos_bboxes.new_full((num_samples,),
                                           -1,
                                           dtype=torch.long)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_gt_inds[:num_pos] = pos_assigned_gt_inds_list
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, bbox_gt_inds
