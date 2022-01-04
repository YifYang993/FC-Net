import os
from saveModel.resultcurve import *
from saveModel.graphgen import *
import shutil


class Visualization(object):
    def __init__(self, opt):
        if not os.path.isdir(opt.save_head_path + opt.save_path):
            os.makedirs(opt.save_head_path + opt.save_path)  
        self.save_path = opt.save_head_path + opt.save_path
        self.wrong_path = self.save_path + "wrongpath.txt"
        self.log_file = self.save_path + "log.txt"
        self.readme = self.save_path + "README.md"
        self.opt_file = self.save_path + "opt.log"
        self.code_path = os.path.join(self.save_path, "code/")
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
        if os.path.isfile(self.readme):
            os.remove(self.readme)
        self.copy_code(dst=self.code_path)
        self.graph = Graph()

    def copy_code(self, src="./", dst="./code/"):
        shutil.copytree(src,
                        dst,
                        ignore=shutil.ignore_patterns('*.pyc', '.csv', '.txt',
                                                      '.json', '.jsonl',
                                                      'log'))

    def writeopt(self, opt):
        with open(self.opt_file, "w") as f:
            for k, v in opt.__dict__.items():
                f.write(str(k) + ": " + str(v) + "\n")

    def writelog(self, input_data):
        print(">>>>> write to log")
        txt_file = open(self.log_file, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def writereadme(self, input_data):
        txt_file = open(self.readme, 'a+')
        txt_file.write(str(input_data) + "\n")
        txt_file.close()

    def drawcurves(self):
        drawer = DrawCurves(file_path=self.log_file, fig_path=self.save_path)
        drawer.draw(target="test_error")
        drawer.draw(target="train_error")

    def gennetwork(self, var):
        self.graph.draw(var=var)

    def savenetwork(self):
        self.graph.save(file_name=self.save_path + "network.svg")
