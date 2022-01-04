import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable

from models.focal_contrastive_Loss import *
from models.Focal_loss_sigmoid import *
from models.modelDifine import *
from saveModel.graphgen import *
from saveModel.save_hyperparameter import *

single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def getlearningrate(epoch, opt):
    # update lr
    lr = opt.LR
    if opt.lrPolicy == "multistep":
        if epoch + 1.0 >= opt.nEpochs * opt.ratio[1]:  # 0.6 or 0.8
            lr = opt.LR * 0.01
        elif epoch + 1.0 >= opt.nEpochs * opt.ratio[0]:  # 0.4 or 0.6
            lr = opt.LR * 0.1
    elif opt.lrPolicy == "linear":
        k = (0.001 - opt.LR) / math.ceil(opt.nEpochs / 2.0)
        lr = k * math.ceil((epoch + 1) / opt.step) + opt.LR
    elif opt.lrPolicy == "exp":
        power = math.floor((epoch + 1) / opt.step)
        lr = lr * math.pow(opt.gamma, power)
    else:
        assert False, "invalid lr policy"

    return lr


def computeAUC(outputs, labels, epoch):
    if isinstance(outputs, list):
        pred = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)
    else:
        pred = outputs
        y = labels
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    if np.isnan(roc_auc):
        roc_auc = 0

    if opt.draw_ROC == True:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(str(epoch) + "epoch_RocOf" + opt.experimentID + ".png")
    return roc_auc, fpr, tpr


def readwrongpath(lists):
    res = []
    for i in lists:
        for j in i:
            res.append(j)
    return res




def computeEval(outputs, labels, pathlist):
    dicts = {}
    wronglist = []
    path = readwrongpath(pathlist)
    if isinstance(outputs, list):
        pred = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)
    else:
        pred = outputs
        y = labels
    prolist = pred.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    for i in range(len(pred)):
        dicts[path[i]] = prolist[i]
        if pred[i] != y[i]:
            wronglist.append(path[i])
    # acc
    acc = metrics.accuracy_score(y, pred)
    # tn, fp, fn, tp
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    # precision
    precision = np.nan if (tp + fp) == 0 else float(tp) / (tp + fp)
    # recall
    recall = np.nan if (tp + fn) == 0 else float(tp) / (tp + fn)
    # F1
    f1 = metrics.f1_score(y, pred, pos_label=1, average='binary')
    # g-mean
    specificity = np.nan if (tn + fp) == 0 else float(tn) / (tn + fp)
    gmean = math.sqrt(recall * specificity)
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)
    return acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist


def printresult(epoch,
                nEpochs,
                count,
                iters,
                lr,
                data_time,
                iter_time,
                loss,
                mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    log_str = ">>> %s [%.3d|%.3d], Iter[%.3d|%.3d], DataTime: %.4f, IterTime: %.4f, lr: %.4f" \
              % (mode, epoch + 1, nEpochs, count, iters, data_time, iter_time, lr)

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time * 0.95 + 0.05 * (data_time +
                                                               iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
    else:
        single_test_time = single_test_time * 0.95 + 0.05 * (data_time +
                                                             iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
    total_time = (single_train_time * single_train_iters +
                  single_test_time * single_test_iters) * nEpochs
    time_str = ",Cost Time: %d Days %d Hours %d Mins %.4f Secs" % (
        total_time // (3600 * 24), total_time // 3600.0 % 24,
        total_time % 3600.0 // 60, total_time % 60)
    print(log_str + time_str)



def generateTarget(images, labels):
    target_disease = torch.LongTensor(labels.size(0)).zero_() + int(1)
    reduce_labels = labels == target_disease
    reduce_labels = reduce_labels.float()
    return reduce_labels



def generate_factor(T, initv=1, T_max=200, power=1):
    a = initv * (1 - (math.pow(float((T / T_max)), power)))
    return a


class Trainer_contra(object):
    realLabelsarr = []
    predictLabelsarr = []

    def __init__(self, model, opt, optimizer=None):
        self.opt = opt
        self.criterion_focal = FocalLoss(alpha=0.75, gamma=2)
        self.model = model
        self.criterion = nn.BCELoss().cuda()
        self.criterion_BCE = nn.BCELoss().cuda()
        self.sigmoid = nn.Sigmoid()
        self.criterion_contra = Focal_ContrastiveLoss()
        self.lr = self.opt.LR
        self.optimzer = optimizer or torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weightDecay,
            nesterov=True)
        self.factor = 0

    def updateopts(self):
        self.optimzer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weightDecay,
                                        nesterov=True)

    def updatelearningrate(self,
                           epoch):  # update learning rate of model optimizer
        self.lr = getlearningrate(epoch=epoch, opt=self.opt)
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def custom_replace(self, tensor, on_zero, on_non_zero):
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def forward(self,
                dark_input_var,
                light_input_var,
                fulldark_var,
                fulllight_var,
                labels_var=None):
        Pair = (dark_input_var, light_input_var, fulldark_var, fulllight_var)
        predict = self.model(Pair)  #h_d, h_l, x
        h_full_d, h_full_l, h_d, h_l, x = predict["h_full_d"], predict[
            "h_full_l"], predict["h_d"], predict["h_l"], predict["x"]
        if labels_var is not None:  ##
            loss0 = self.criterion_focal(x, labels_var)
            loss1 = self.criterion_contra(
                h_d, h_l, labels_var) + 0.5 * self.criterion_contra(
                    h_full_d, h_full_l, labels_var)

        else:
            loss = None
        return x, loss0, loss1




    def backward(self, loss):
        self.optimzer.zero_grad()
        loss.backward()

        self.optimzer.step()

    def train(self, epoch, train_loader):
        loss_sum = 0
        iters = len(train_loader)
        output_list = []
        label_list = []
        self.updatelearningrate(epoch)

        self.model.train()

        start_time = time.time()
        end_time = start_time
        self.factor = generate_factor(T=epoch)
        for i, (dark_input, light_input, labels, _) in enumerate(train_loader):
            self.model.train()
            start_time = time.time()
            data_time = start_time - end_time
            """
            label for two branch
            """
            labels_synechia = generateTarget(dark_input[0], labels)
            reduce_labels_synechia = labels_synechia
            labels_synechia = labels_synechia.cuda(non_blocking=True)
            labels_var = Variable(labels_synechia)

            dark_var = Variable(dark_input[0].cuda(non_blocking=True))
            light_var = Variable(light_input[0].cuda(non_blocking=True))
            dark_full_var = Variable(dark_input[1].cuda(non_blocking=True))
            light_full_var = Variable(light_input[1].cuda(non_blocking=True))
            output, loss0, loss1 = self.forward(
                dark_var, light_var, dark_full_var, light_full_var,
                labels_var)  ##loss0 focal loss1 contra
            prediction = output.data.cpu()
            output_list.append(prediction.numpy())
            label_list.append(reduce_labels_synechia.cpu().numpy())

            loss = loss0 + self.factor * loss1
            self.backward(loss)
            loss_sum += float(loss.data)
            end_time = time.time()

            iter_time = end_time - start_time

        loss_sum /= iters
        auc, fpr, tpr = computeAUC(output_list, label_list, epoch)
        print("|===>Training AUC: %.4f Loss: %.4f " % (auc, loss_sum))
        if not os.path.exists(
                os.path.join(opt.save_head_path, opt.save_path, "model/",
                             "hyperparameter.json")):
            print("save hyperparameter")
            self.save_hyper.dump()
        return auc, loss_sum

    def test(self, epoch, test_loader):
        loss_sum = 0
        iters = len(test_loader)
        output_list = []
        label_list = []
        pathlist = []
        self.model.eval()
        print("testing", self.factor)
        start_time = time.time()
        end_time = start_time
        for i, (dark_input, light_input, labels,
                image_name) in enumerate(test_loader):
            with torch.no_grad():
                print(i)
                pathlist.append(image_name)
                start_time = time.time()
                data_time = start_time - end_time

                labels_synechia = generateTarget(dark_input[0], labels)
                reduce_labels_synechia = labels_synechia

                labels_synechia = labels_synechia.cuda(non_blocking=True)
                labels_var = Variable(labels_synechia)
                labels_var = Variable(labels_var)

                dark_var = Variable(dark_input[0].cuda(non_blocking=True))
                light_var = Variable(light_input[0].cuda(non_blocking=True))
                dark_full_var = Variable(dark_input[1].cuda(non_blocking=True))
                light_full_var = Variable(
                    light_input[1].cuda(non_blocking=True))
                output, loss0, loss1 = self.forward(dark_var, light_var,
                                                    dark_full_var,
                                                    light_full_var, labels_var)
                loss = loss0 + self.factor * loss1
                loss_sum += float(loss.data) / iters
                prediction = output.data.cpu()
                output_list.append(prediction.numpy())
                label_list.append(reduce_labels_synechia.cpu().numpy())
                end_time = time.time()
                iter_time = end_time - start_time

        auc, fpr, tpr = computeAUC(output_list, label_list, epoch)
        acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist = computeEval(
            output_list, label_list, pathlist)
        print(
            "|===>Testing AUC: %.4f Loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f gmean: %.4f"
            % (auc, loss_sum, acc, precision, recall, f1, gmean))
        return auc, loss_sum, acc, precision, recall, f1, gmean, tn, fp, fn, tp, wronglist
