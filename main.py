import torch.backends.cudnn as cudnn
from termcolor import colored

from dataloader.asoct_dataloader import *
from opt.opt import *
from saveModel.checkpoint import *
from saveModel.save_hyperparameter import *
from trainer import *
from visualization import *


def dataparallel(model, gpulist):
    ngpus = len(gpulist)
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = gpulist
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


def getweights(layer, epoch_id, block_id, layer_id, log_writer):
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu().numpy()
        weights_view = weights.reshape(weights.size)
        log_writer(input_data=weights_view,
                   block_id=block_id,
                   layer_id=layer_id,
                   epoch_id=epoch_id)


def main(net_opt=None):
    """requirements:
    apt-get install graphviz
    pip install pydot termcolor"""
    opt = net_opt or NetOption()

    start_time = time.time()

    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    if len(opt.gpulist) == 1 and torch.cuda.device_count() >= 1:
        assert opt.opt.gpulist[0] <= torch.cuda.device_count(
        ) - 1, "Invalid GPU ID"
        torch.cuda.set_device(opt.gpulist[0])
    else:
        torch.cuda.set_device(opt.gpulist[0])

    # create data loader
    data_loader = DataLoader(dataset=opt.data_set,
                             data_path=opt.data_path,
                             label_path=opt.label_path,
                             batch_size=opt.batchSize,
                             rootpath=opt.rootpath,
                             n_threads=opt.nThreads,
                             ten_crop=opt.tenCrop,
                             dataset_ratio=opt.datasetRatio)
    train_loader, test_loader = data_loader.getloader()

    # define check point
    check_point = CheckPoint(opt=opt)
    # create residual network mode

    if opt.resume:
        check_point_params = check_point.resumemodel()
    else:
        check_point_params = check_point.check_point_params

    try:
        optimizer = check_point_params['opts']
        start_stage = check_point_params['stage']
        start_epoch = check_point_params['resume_epoch']
        if check_point_params['resume_epoch'] is not None:
            start_epoch += 1
        if start_epoch >= opt.nEpochs:
            start_epoch = 0
            start_stage += 1
    except:
        optimizer = None
        start_stage = None
        start_epoch = None

    # model

    if opt.netType == "I3D":
        model = InceptionI3d()
        print("loading pretrain model")
        mydict = model.state_dict()
        state_dict = torch.load("./pretrained/weights/rgb_imagenet.pt",
                                map_location=torch.device('cpu'))
        pretrained_dict_l = {}
        print("loading for large module")
        for k, v in state_dict.items():
            if k in [
                    "fc8.bias", "logits.conv3d.weight", "logits.conv3d.bias",
                    "fc8.weight"
            ]:
                continue
            splits = k.split(".")
            splits[0] = splits[0] + "_l"
            pretrained_dict_l[".".join(splits)] = v
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k not in ["fc8.bias", "logits.conv3d.weight", "fc8.weight"]
        }
        mydict.update(pretrained_dict)
        mydict.update(pretrained_dict_l)
        model.load_state_dict(mydict)
        model.replace_logits(1)
    if opt.netType == "resnet3d":
        model = resnet3d(num_classes=opt.numclass, use_nl=True)
        if opt.resume and opt.pretrain:
            print("error: loading two pretrain model")
            return
        if opt.resume:
            state_dict = check_point_params["model"]
            model.load_state_dict(state_dict)
        if opt.pretrain:
            mydict = model.state_dict()
            state_dict = torch.load(opt.pretrain)
            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in ["fc.bias", 'fc.weight']
            }
            mydict.update(pretrained_dict)
            model.load_state_dict(mydict)

    model = dataparallel(model, opt.gpulist)
    trainer = Trainer_contra(model=model, opt=opt, optimizer=optimizer)
    print("|===>Create trainer")

    if opt.testOnly:
        trainer.test(epoch=0, test_loader=test_loader)
        return

    # define visualizer
    visualize = Visualization(opt=opt)
    visualize.writeopt(opt=opt)
    best_auc = 0
    start_epoch = opt.resumeEpoch
    for epoch in range(start_epoch, opt.nEpochs):
        train_auc, train_loss = trainer.train(epoch=epoch,
                                              train_loader=train_loader)
        test_auc, test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp, wronglist = trainer.test(
            epoch=epoch, test_loader=test_loader)
        # write and print result
        log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t" % (
            epoch, train_auc, train_loss, test_auc, test_loss, test_acc,
            test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp)
        visualize.writelog(log_str)
        best_flag = False
        if best_auc <= test_auc:
            best_auc = test_auc
            best_flag = True
            print(
                colored(
                    "# %d ==>Best Result is: AUC: %f\n" % (epoch, best_auc),
                    "red"))
        else:
            print(
                colored(
                    "# %d ==>Best Result is: AUC: %f\n" % (epoch, best_auc),
                    "blue"))

        # save check_point
        check_point.savemodel(epoch=epoch,
                              model=trainer.model,
                              opts=trainer.optimzer,
                              best_flag=best_flag)

    end_time = time.time()
    time_interval = end_time - start_time
    t_hours = time_interval // 3600
    t_mins = time_interval % 3600 // 60
    t_sec = time_interval % 60
    t_string = "Running Time is: " + \
               str (t_hours) + " hours, " + str (t_mins) + \
               " minutes," + str (t_sec) + " seconds\n"
    print(t_string)


if __name__ == '__main__':
    main_opt = NetOption()
    print(">>>>running experiment:", main_opt.experimentID)
    main_opt.paramscheck()
    main(main_opt)
