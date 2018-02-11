import os
import sys
import time
import torch
import cv2
from darknet.utils.defaults import argHandler
from darknet.nets.yolo2 import Darknet
from darknet.datasets import dataset
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
from darknet.utils.cfg import parse_cfg
from darknet.utils.bbox import nms,get_region_boxes,bbox_iou
from darknet.utils.bbox import do_detect,load_class_names,plot_boxes_cv2,get_image_size
from scripts.voc_eval import do_eval

def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count(b'\n')
    thefile.close( )
    return count


def adjust_learning_rate(lr, scales, steps, batch_size, optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr



def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this):
                os.makedirs(this)

    _get_dir([FLAGS.backup])


    net_options = parse_cfg(FLAGS.config)[0]
    batch_size = int(net_options['batch'])
    max_batches = int(net_options['max_batches'])
    learning_rate = float(net_options['learning_rate'])
    momentum = float(net_options['momentum'])
    decay = float(net_options['decay'])
    steps = [float(step) for step in net_options['steps'].split(',')]
    scales = [float(scale) for scale in net_options['scales'].split(',')]

    # Train parameters
    nsamples = file_lines(FLAGS.trainlist)
    max_epochs = max_batches * batch_size // nsamples + 1
    use_cuda = True
    seed = int(time.time())
    eps = 1e-5
    save_interval = 10  # epoches

    # Test parameters
    conf_thresh = 0.25
    nms_thresh = 0.4
    iou_thresh = 0.5
    num_workers=10

    torch.manual_seed(seed)
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
        torch.cuda.manual_seed(seed)

    model = Darknet(FLAGS.config)
    region_loss = model.loss

    model.load_weights(FLAGS.binary)
    model.print_network()

    region_loss.seen = model.seen
    processed_batches = model.seen//batch_size

    ngpus = len(FLAGS.gpus.split(','))


    init_width = model.width
    init_height = model.height
    init_epoch = model.seen // nsamples

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(FLAGS.testlist, shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)
    if use_cuda:
        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if key.find('.bn') >= 0 or key.find('.bias') >= 0:
            params += [{'params': [value], 'weight_decay': 0.0}]
        else:
            params += [{'params': [value], 'weight_decay': decay * batch_size}]

    optimizer = optim.SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                          weight_decay=decay * batch_size)


    def test():
        def truths_length(truths):
            for i in range(50):
                if truths[i][1] == 0:
                    return i

        model.eval()
        if ngpus > 1:
            cur_model = model.module
        else:
            cur_model = model
        num_classes = cur_model.num_classes
        anchors = cur_model.anchors
        num_anchors = cur_model.num_anchors
        total = 0.0
        proposals = 0.0
        correct = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = model(data).data
            all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
            for i in range(output.size(0)):
                boxes = all_boxes[i]
                boxes = nms(boxes, nms_thresh)
                truths = target[i].view(-1, 5)
                num_gts = truths_length(truths)

                total = total + num_gts

                for i in range(len(boxes)):
                    if boxes[i][4] > conf_thresh:
                        proposals = proposals + 1

                for i in range(num_gts):
                    box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                    best_iou = 0
                    best_j = -1
                    for j in range(len(boxes)):
                        iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou
                    if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                        correct = correct + 1

        precision = 1.0 * correct / (proposals + eps)
        recall = 1.0 * correct / (total + eps)
        fscore = 2.0 * precision * recall / (precision + recall + eps)
        logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))

    def detect(imgfile):
        if ngpus > 1:
            cur_model = model.module
        else:
            cur_model = model
        cur_model.print_network()
        cur_model.load_weights(FLAGS.binary)
        cur_model.cuda()

        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (cur_model.width, cur_model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for i in range(2):
            start = time.time()
            boxes = do_detect(cur_model, sized, 0.5, 0.4)
            stop = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (stop - start)))

        class_names = load_class_names(FLAGS.namesFile)
        plot_boxes_cv2(img, boxes, savename='./data/out/predictions.jpg', class_names=class_names)

    def valid(outfile):
        valid_images = FLAGS.testlist
        name_list = FLAGS.namesFile
        prefix = 'data/testResult'
        names = load_class_names(name_list)

        with open(valid_images) as fp:
            tmp_files = fp.readlines()
            valid_files = [item.rstrip() for item in tmp_files]

        if ngpus > 1:
            cur_model = model.module
        else:
            cur_model = model
        cur_model.print_network()
        cur_model.load_weights(FLAGS.binary)
        cur_model.cuda()
        cur_model.eval()

        valid_dataset = dataset.listDataset(valid_images, shape=(cur_model.width, cur_model.height),
                                            shuffle=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                            ]))
        valid_batchsize = 2
        assert (valid_batchsize > 1)

        kwargs = {'num_workers': 4, 'pin_memory': True}
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)

        fps = [0] * cur_model.num_classes
        for i in range(cur_model.num_classes):
            buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
            fps[i] = open(buf, 'w')

        lineId = -1

        conf_thresh = 0.005
        nms_thresh = 0.45
        for batch_idx, (data, target) in enumerate(valid_loader):
            data = data.cuda()
            data = Variable(data, volatile=True)
            output = cur_model(data).data
            batch_boxes = get_region_boxes(output, conf_thresh, cur_model.num_classes, cur_model.anchors, cur_model.num_anchors, 0, 1)
            for i in range(output.size(0)):
                lineId = lineId + 1
                fileId = os.path.basename(valid_files[lineId]).split('.')[0]
                width, height = get_image_size(valid_files[lineId])
                print(valid_files[lineId])
                boxes = batch_boxes[i]
                boxes = nms(boxes, nms_thresh)
                for box in boxes:
                    x1 = (box[0] - box[2] / 2.0) * width
                    y1 = (box[1] - box[3] / 2.0) * height
                    x2 = (box[0] + box[2] / 2.0) * width
                    y2 = (box[1] + box[3] / 2.0) * height

                    det_conf = box[4]
                    for j in range((len(box) - 5) // 2):
                        cls_conf = box[5 + 2 * j]
                        cls_id = box[6 + 2 * j]
                        prob = det_conf * cls_conf
                        fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))

        for i in range(cur_model.num_classes):
            fps[i].close()

    if FLAGS.train:
        for epoch in range(init_epoch, max_epochs):
            t0 = time.time()
            if ngpus > 1:
                cur_model = model.module
            else:
                cur_model = model
            train_loader = torch.utils.data.DataLoader(
                dataset.listDataset(FLAGS.trainlist, shape=(init_width, init_height),
                                    shuffle=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]),
                                    train=True,
                                    seen=cur_model.seen,
                                    batch_size=batch_size,
                                    num_workers=num_workers),
                batch_size=batch_size, shuffle=False, **kwargs)

            lr = adjust_learning_rate(learning_rate, scales, steps, batch_size, optimizer, processed_batches)
            logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
            model.train()
            t1 = time.time()
            avg_time = torch.zeros(9)
            for batch_idx, (data, target) in enumerate(train_loader):
                t2 = time.time()
                adjust_learning_rate(learning_rate, scales, steps, batch_size, optimizer, processed_batches)
                processed_batches = processed_batches + 1
                if use_cuda:
                    data = data.cuda()

                t3 = time.time()
                data, target = Variable(data), Variable(target)
                t4 = time.time()
                optimizer.zero_grad()
                t5 = time.time()
                output = model(data)
                t6 = time.time()
                region_loss.seen = region_loss.seen + data.data.size(0)
                loss = region_loss(output, target)
                t7 = time.time()
                loss.backward()
                t8 = time.time()
                optimizer.step()
                t9 = time.time()
                if False and batch_idx > 1:
                    avg_time[0] = avg_time[0] + (t2 - t1)
                    avg_time[1] = avg_time[1] + (t3 - t2)
                    avg_time[2] = avg_time[2] + (t4 - t3)
                    avg_time[3] = avg_time[3] + (t5 - t4)
                    avg_time[4] = avg_time[4] + (t6 - t5)
                    avg_time[5] = avg_time[5] + (t7 - t6)
                    avg_time[6] = avg_time[6] + (t8 - t7)
                    avg_time[7] = avg_time[7] + (t9 - t8)
                    avg_time[8] = avg_time[8] + (t9 - t1)
                    print('-------------------------------')
                    print('       load data : %f' % (avg_time[0] / (batch_idx)))
                    print('     cpu to cuda : %f' % (avg_time[1] / (batch_idx)))
                    print('cuda to variable : %f' % (avg_time[2] / (batch_idx)))
                    print('       zero_grad : %f' % (avg_time[3] / (batch_idx)))
                    print(' forward feature : %f' % (avg_time[4] / (batch_idx)))
                    print('    forward loss : %f' % (avg_time[5] / (batch_idx)))
                    print('        backward : %f' % (avg_time[6] / (batch_idx)))
                    print('            step : %f' % (avg_time[7] / (batch_idx)))
                    print('           total : %f' % (avg_time[8] / (batch_idx)))
                t1 = time.time()
            print('')
            t1 = time.time()
            logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
            if (epoch + 1) % save_interval == 0:
                logging('save weights to %s/%06d.weights' % (FLAGS.backup, epoch + 1))
                cur_model.seen = (epoch + 1) * len(train_loader.dataset)
                cur_model.save_weights('%s/%06d.weights' % (FLAGS.backup, epoch + 1))

            test()

    elif FLAGS.eval:
        valid( 'comp_test_')
        do_eval('data/voc/VOCdevkit', 'data/testResult/comp_test_', 'data/output')

    elif FLAGS.demo:
        detect(FLAGS.testImage)