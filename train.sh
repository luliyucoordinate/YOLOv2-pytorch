#!/bin/sh
python cvs.py --config cfg/yolo-voc.cfg --binary data/weights/darknet19_448.conv.23 --trainlist data/voc/train.txt -- testlist data/voc/2007_test.txt