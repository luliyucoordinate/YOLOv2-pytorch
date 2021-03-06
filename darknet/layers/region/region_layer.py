import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from darknet.build import *


class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.nobject_scale = 1
        self.bject_scale = 1
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        t0 = time.time()
        #output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB,_,nH,nW = output.size()
        nA = self.num_anchors
        nC = self.num_classes

        output = output.view(nB, nA, (5 + nC), nH, nW)
        coordinate_x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        coordinate_y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        coordinate_w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        coordinate_h = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        confidence = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1), nC).long().cuda())
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = coordinate_x.data + grid_x
        pred_boxes[1] = coordinate_y.data + grid_y
        pred_boxes[2] = torch.exp(coordinate_w.data) * anchor_w
        pred_boxes[3] = torch.exp(coordinate_h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes,
                                                                                                    target.data,
                                                                                                    self.anchors, nA,
                                                                                                    nC,
                                                                                                    nH, nW,
                                                                                                    self.noobject_scale,
                                                                                                    self.object_scale,
                                                                                                    self.thresh,
                                                                                                    self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((confidence > 0.25).sum().data[0])

        tx = Variable(tx.cuda())
        ty = Variable(ty.cuda())
        tw = Variable(tw.cuda())
        th = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask = Variable(conf_mask.cuda().sqrt())
        cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
        cls = cls[cls_mask].view(-1, nC)

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(coordinate_x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(coordinate_y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(coordinate_w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(coordinate_h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(confidence * conf_mask, tconf * conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0],
        loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss