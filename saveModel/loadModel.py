import torch
from opt.opt import *
from models.model_mscale import *

def loadit(check_point_params, model):
    # print(model)
    model.conv1 = check_point_params.conv1
    model.conv1_m = check_point_params.conv1_m
    model.conv11 = check_point_params.conv11
    model.conv11_m = check_point_params.conv11_m
    model.bn1 = check_point_params.bn1
    model.bn1_m = check_point_params.bn1_m
    model.relu = check_point_params.relu
    model.relu_m = check_point_params.relu_m
    model.maxpool1 = check_point_params.maxpool1
    model.maxpool1_m = check_point_params.maxpool1_m
    model.maxpool2 = check_point_params.maxpool2
    model.maxpool2_m = check_point_params.maxpool2_m
    model.conv11_final = check_point_params.conv11_final
    model.layer1 = check_point_params.layer1
    model.layer2 = check_point_params.layer2
    model.layer3 = check_point_params.layer3
    model.layer4 = check_point_params.layer4
    model.layer1_m = check_point_params.layer1_m
    model.layer2_m = check_point_params.layer2_m
    model.layer3_m = check_point_params.layer3_m
    model.layer4_m = check_point_params.layer4_m
    model.avgpool = check_point_params.avgpool
    model.avgpool_m = check_point_params.avgpool_m
    model.fc = check_point_params.fc
    print(check_point_params)


def main():
    # opt = NetOption
    xl = Variable(torch.randn(4, 21, 3, 244, 244).cpu())
    xd = Variable(torch.randn(4, 21, 3, 244, 244).cpu())
    fullxl = Variable(torch.randn(4, 21, 3, 244, 244).cpu())
    fullxd = Variable(torch.randn(4, 21, 3, 244, 244).cpu())
    x = (xl, xd, fullxd, fullxl)


    resume = "/mnt/dataset/model/darklight/log_asoct_resnet3d_18_onevsall_bs4_multiscale_light_resnet3d_0310/model/best_model.pkl"
    check_point_params = torch.load(resume)["model"]
    model = resnet3d().cpu()
    a = model(x)
    print(a)
    loadit(check_point_params, model)
    b = model(x)
    print(b)



if __name__ == '__main__':
    main()