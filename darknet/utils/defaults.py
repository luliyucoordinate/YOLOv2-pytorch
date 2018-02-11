import sys
import argparse
import numpy as np
import time

class argHandler(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h':'show this super helpful message and exit'}

    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Example usage: flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/yolo.weights')
        print('')
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        print('')
        exit()

    def parseArgs(self, args):
        print('')
        i = 1
        while i < len(args):
            if args[i] == '-h' or args[i] == '--h' or args[i] == '--help':
                self.help() #Time for some self help! :)
            if len(args[i]) < 2:
                print('ERROR - Invalid argument: ' + args[i])
                print('Try running flow --help')
                exit()
            argumentName = args[i][2:]
            if isinstance(self.get(argumentName), bool):
                if not (i + 1) >= len(args) and (args[i + 1].lower() != 'false' and args[i + 1].lower() != 'true') and not args[i + 1].startswith('--'):
                    print('ERROR - Expected boolean value (or no value) following argument: ' + args[i])
                    print('Try running flow --help')
                    exit()
                elif not (i + 1) >= len(args) and (args[i + 1].lower() == 'false' or args[i + 1].lower() == 'true'):
                    self[argumentName] = (args[i + 1].lower() == 'true')
                    i += 1
                else:
                    self[argumentName] = True
            elif args[i].startswith('--') and not (i + 1) >= len(args) and not args[i + 1].startswith('--') and argumentName in self:
                if isinstance(self[argumentName], float):
                    try:
                        args[i + 1] = float(args[i + 1])
                    except:
                        print('ERROR - Expected float for argument: ' + args[i])
                        print('Try running flow --help')
                        exit()
                elif isinstance(self[argumentName], int):
                    try:
                        args[i + 1] = int(args[i + 1])
                    except:
                        print('ERROR - Expected int for argument: ' + args[i])
                        print('Try running flow --help')
                        exit()
                self[argumentName] = args[i + 1]
                i += 1
            else:
                print('ERROR - Invalid argument: ' + args[i])
                print('Try running flow --help')
                exit()
            i += 1

    def setDefaults(self):

        self.define('config', './cfg/yolo-voc.cfg', 'path to .cfg directory')
        self.define('binary', './data/weights/yolo-voc.weights', 'path to .weights directory')
        self.define('load', '', ' from .weights')
        self.define('testImage', './data/img/dog.jpg', 'testImage for demo')
        self.define('namesFile', './data/voc.names', 'namesfile for dataset')

        self.define('trainlist', 'D:/voc_train.txt', 'path to trainlist')
        self.define('testlist', 'D:/2007_test.txt', 'path to testlist')
        self.define('gpu', True, 'use gpu to train')
        self.define('gpus', '0', 'how much gpu (from 0.0 to 1.0)')
        self.define('backup', './backup/', 'path to backup folder')

        self.define('summary', './summary/', 'path to TensorBoard summaries directory')
        self.define('train', False, 'train the whole net')
        self.define('demo', False, 'demo')
        self.define('eval', False, 'eval the whole net')

