import os
import math


class NetOption(object):
    def __init__(self):
        #  ------------ General options ---------------------------------------
        self.save_head_path = "/mnt/cephfs/home/yangyifan/yangyifan/model/synechiae/"  # where to save model and log code etc#/home/yangyifan/model/synechiae/#/mnt/dataset/model/darklight/
        self.data_path = "/mnt/cephfs/home/yangyifan/code/multiViewCNN/Multi_ViewCNN/dataProcess/"  # path for loading data set  \
        self.label_path = "/mnt/cephfs/home/yangyifan/yangyifan/code/multiviewCNN_quarter/multiViewCNN/Multi_ViewCNN/dataProcess/add_alone_clock/one_for_new_clock_labelv1.csv"  ###one lable cut in half 0413
        self.rootpath = "/mnt/cephfs/dataset/medical/splited_Casia2"
        self.data_set = "asoct"  # options: asoct
        self.disease_type = 1  # 1(open) | 2(narrow) | 3(close) | 4(unclassify)  or  1(open) | 2(narrow/close)
        self.manualSeed = 1  # manually set RNG seed
        self.gpulist = [2, 6]
        self.datasetRatio = 1.0  # greedy increasing training data for cifar10
        self.numclass = 1
        # ------------- Data options ------------------------------------------
        self.nThreads = 10  # number of data loader threads
        self.dataset = "internal"  # BJ | internal
        self.imgsize = 244
        # ------------- Training options --------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options -------------------------------------
        self.nEpochs = 200  # number of total epochs to train
        self.batchSize = 8  # mini-batch size
        self.LR = 0.001  # initial learning rate
        self.lrPolicy = "multistep"  # options: multistep | linear | exp
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-2
        self.gamma = 0.94  # gamma for learning rate policy (step)
        self.step = 2.0  # step for learning rate policy

        # ---------- Model options --------------------------------------------
        ###--draw Roc---###
        self.draw_ROC = False
        # ---------- Model options --------------------------------------------
        self.trainingType = 'onevsall'  
        self.netType = "resnet3d"  # options: | I3D | resnet3d 
        self.experimentID = "baseline"  

        # ---------- Resume or Retrain options --------------------------------
        self.resume_I3D = False
        self.retrain = None  # path to model to retrain with
        self.resume = None  # path to directory containing checkpoint
        self.resumeEpoch = 0  # manual epoch number for resume
        self.pretrain = None
        self.resume = None
        self.pretrain = "/mnt/cephfs/home/yangyifan/yangyifan/232/yangyifan/code/multiViewCNN/pretrained/i3d_r50_nl_kinetics.pth"
        # check parameters
        self.paramscheck()

    def paramscheck(self):
        self.save_path = "log_%s_%s_bs%d_%s/" % \
                         (self.data_set, self.netType,self.batchSize, self.experimentID)
        if self.data_set == 'asoct':
            self.nClasses = 1
            self.ratio = [1.0 / 2, 2.7 / 3]
