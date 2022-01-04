
import json
import os
class save_hyperparameter():
    def __init__(self, rootpath):
        self.json_path=os.path.join(rootpath, "hyperparameter.json")
        self.temp_dict={}

    def write_to_dict(self, key, value):
        if key not in self.temp_dict:
            self.temp_dict[key] = value
        else:
            print("save key exist!", key)
    def dump(self):
        result=[]
        result.append(self.temp_dict)
        with open(self.json_path, 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    savehead=save_hyperparameter("/home/datasets/CASIA2/model/darklight/log_asoct_resnet3d_18_onevsall_bs8__resnet3d_alpha0.25_gamma1_m1_focal_contra_reduce_ratio=4_119/")
    savehead.write_to_dict("margin",2)
    savehead.write_to_dict ("gamma", 3)
    savehead.write_to_dict ("alpha", 20)
    savehead.dump()

