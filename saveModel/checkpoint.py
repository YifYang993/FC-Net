import torch
import os

# note:
'''
# save and load entire model
torch.save(model, "model.pkl")
model = torch.load("model.pkl")
# save and load only the model parameters(recommended)
torch.save(model.state_dict(), "params.pkl")
model.load_state_dict(torch.load("params.pkl"))
'''


class CheckPoint(object):
    def __init__(self, opt):
        self.resume = opt.resume
        self.nGPU = len(opt.gpulist)
        self.resumeEpoch = opt.resumeEpoch
        self.retrain = opt.retrain
        self.save_path = opt.save_head_path+opt.save_path+"model/"
        self.check_point_params = {'model': None,
                                   'opts': None,
                                   'stage': None,
                                   'resume_epoch': None}

    def retrainmodel(self):
        if os.path.isfile(self.retrain):
            print ("|===>Retrain model from:", self.retrain)
            retrain_data = torch.load(self.retrain)
            self.check_point_params['model'] = retrain_data['model']
            return self.check_point_params
        else:
            assert False, "file not exits"

    def resumemodel(self):
        if os.path.isfile(self.resume):
            print ("|===>Resume check point from:", self.resume)
            self.check_point_params = torch.load(self.resume)
            if self.resumeEpoch != 0:
                self.check_point_params['resume_epoch'] = self.resumeEpoch
            return self.check_point_params
        else:
            assert False, "file not exits"

    def savemodel(self, epoch=None, model=None, opts=None, best_flag=False):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if self.nGPU >1:
            self.check_point_params['model'] = model.module.state_dict()
            self.check_point_params['opts'] = opts
            self.check_point_params['resume_epoch'] = epoch

            # torch.save(self.check_point_params, self.save_path+"checkpoint" + str(epoch) +".pkl")
            # torch.save(model, self.save_path + "checkpoint" + str(epoch) + ".pkl")
            if best_flag:
                best_model = {'model': model.module.state_dict()}
                torch.save(best_model, self.save_path+"best_model.pkl")
        else:
            self.check_point_params['model'] = model.state_dict()
            self.check_point_params['opts'] = opts
            self.check_point_params['resume_epoch'] = epoch

            # torch.save(self.check_point_params, self.save_path + "checkpoint" + str(epoch) + ".pkl")
            # torch.save(model, self.save_path + "checkpoint" + str(epoch) + ".pkl")
            if best_flag:
                best_model = {'model': model.state_dict()}
                torch.save(best_model, self.save_path + "best_model.pkl")
