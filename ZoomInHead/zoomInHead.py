import glob
import os
import random

import cv2
import numpy as np
from PIL import Image


def cal_vertical_center(image_list, savepath):
    image_column_sum_l = 0
    image_column_sum_r = 0
    for image_file in image_list:
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        image_l = image[0:h, 0:w // 2]
        image_r = image[0:h, w // 2:w]
        image_column_sum_l += np.sum(image_l, axis=1)
        image_column_sum_r += np.sum(image_r, axis=1)
    crop_vertical_center_l = np.argmax(image_column_sum_l)
    crop_vertical_center_r = np.argmax(image_column_sum_r)
    return crop_vertical_center_r, crop_vertical_center_l


class VerticalCrop(object
                   ):  ####only crop left part   we should add a right part
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.vertical_center = None
        self.ratio = 0.05
        self.hwr = 0.6

    def __call__(self, img):
        # print("!!", self.vertical_center)
        image_width = img.size[0]
        image_height = img.size[1]
        vertical_center = self.vertical_center + int(self.ratio * image_width)
        crop_height = image_width * self.hwr
        if vertical_center - crop_height // 2 < 0:
            x1 = 0
            y1 = 0
            x2 = image_width
            y2 = crop_height
        elif vertical_center + crop_height // 2 > image_height:
            x1 = 0
            y1 = image_height - crop_height
            x2 = image_width
            y2 = image_height
        else:
            x1 = 0
            y1 = vertical_center - crop_height // 2
            x2 = image_width
            y2 = vertical_center + crop_height // 2

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self, vertical_center):
        """
        custom_extend: vertical center coordinate for estimated spur sceleral position.
        """
        self.vertical_center = vertical_center


class RandomGammaCorrection():
    def __init__(self):
        self.gamma = 1.0

    def __call__(self, img):
        img = np.asarray(img)
        img = np.power(img / 255.0, self.gamma)
        img = np.uint8(img * 255.0)

        return Image.fromarray(img)

    def randomize_parameters(self, custom_extend=None):
        self.gamma = np.random.uniform(1, 2, 1)
        if random.random() < 0.5:
            self.gamma = 1 / self.gamma


if __name__ == '__main__':
    root_list = ""
    target_path = ""

    all_image_path = sorted(glob.glob(os.path.join(root_list, '*.jpg')))

    for single_img_path in all_image_path:

        frontpath, filename = os.path.split(single_img_path)
        image_list = [single_img_path]

        vertical_center_r, vertical_center_l = cal_vertical_center(
            image_list, single_img_path)

        vrc = VerticalCrop(256)
        rgc = RandomGammaCorrection()
        rgc.randomize_parameters()
        vrc.randomize_parameters(vertical_center_r)
        image = Image.open(image_list[0])
        w, h = image.size
        image_r = image.crop((w // 2, 0, w, h))
        crop_image_r = np.asarray(vrc(image_r))
        crop_image_r = np.asarray(rgc(crop_image_r))

        image_l = image.crop((0, 0, w // 2, h))
        crop_image_l = np.asarray(vrc(image_l))
        crop_image_l = np.asarray(rgc(crop_image_l))

        cv2.imwrite(os.path.join(target_path, "right" + filename),
                    crop_image_r)

        vrc_l = VerticalCrop(256)
        rgc_l = RandomGammaCorrection()
        rgc_l.randomize_parameters()
        vrc_l.randomize_parameters(vertical_center_l)
        image_l = image.crop((0, 0, w // 2, h))
        crop_image_l = np.asarray(vrc_l(image_l))
        crop_image_l = np.asarray(rgc_l(crop_image_l))
        cv2.imwrite(os.path.join(target_path, "left" + filename), crop_image_l)
