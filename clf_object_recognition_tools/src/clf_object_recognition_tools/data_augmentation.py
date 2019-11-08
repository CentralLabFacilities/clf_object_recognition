#!/usr/bin/env python
import cv2

import os
import copy
import imghdr
import argparse

import numpy as np
from scipy import ndimage
import random
from random import randint

import utils


def readable_dir(prospective_dir):
    if not os.path.isdir(prospective_dir):
        raise Exception("readable_dir:{0} is not a valid path".format(prospective_dir))
    elif os.access(prospective_dir, os.R_OK):
        return prospective_dir
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(prospective_dir))


def valid_dir(prospective_dir):
    if not os.path.exists(prospective_dir):
        os.makedirs(prospective_dir)

    if not os.path.isdir(prospective_dir):
        raise Exception("readable_dir:{0} is not a valid path".format(prospective_dir))
    elif os.access(prospective_dir, os.R_OK):
        return prospective_dir
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(prospective_dir))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def change_illumination(image, new_path, min_gamma, max_gamma):
    max_image = np.zeros(image.shape) + 255  # makes sure the values won't be higher then 255
    lf = random.uniform(min_gamma, max_gamma)
    tmp_image = adjust_gamma(image.copy(), gamma=lf)
    tmp_image = tmp_image.astype(int)
    tmp_image = np.fmin(tmp_image, max_image)
    tmp_path = new_path[0:(len(new_path) - 4)] + "_l" + str(lf).replace('.', '') + new_path[(len(new_path) - 4):]
    return tmp_image, tmp_path


def mirror_image(image, bboxes, new_path):
    tmp_image = cv2.flip(image, 1)

    tmp_bboxes = []
    for bbox in bboxes:
        tmp_bbox = copy.deepcopy(bbox)
        # swap x_center
        tmp_bbox.x_center = abs(bbox.x_center - 1)
        tmp_bboxes.append(tmp_bbox)

    tmp_path = new_path[0:(len(new_path) - 4)] + "_m" + new_path[(len(new_path) - 4):]
    return tmp_image, tmp_path, tmp_bboxes


def change_scale(scale_image, bboxes, new_path, min_scale, max_scale):
    sf = random.uniform(min_scale, max_scale)
    tmp_image = scale_image.copy()

    h, w, _ = tmp_image.shape
    tmp_image = cv2.resize(tmp_image, (0,0), fx=sf, fy=sf)
    new_h, new_w, _ = tmp_image.shape
    offset_h = int((h - new_h) / 2)
    offset_w = int((w - new_w) / 2)
    tmp_image = np.pad(tmp_image, ((offset_h, offset_h + (new_h % 2)), (offset_w, offset_w + (new_w % 2)), (0, 0)),
                       'constant', constant_values=255)

    tmp_bboxes = []
    for bbox in bboxes:
        tmp_bbox = copy.deepcopy(bbox)
        tmp_bbox.x_center = (float(offset_w) + (bbox.x_center*float(new_w)))/float(w)
        tmp_bbox.y_center = (float(offset_h) + (bbox.y_center*float(new_h)))/float(h)
        tmp_bbox.width = (bbox.width*float(new_w))/float(w)
        tmp_bbox.height = (bbox.height*float(new_h))/float(h)
        tmp_bboxes.append(tmp_bbox)

    tmp_path = new_path[0:(len(new_path) - 4)] + "_s" + str(sf).replace('.', '') + new_path[(len(new_path) - 4):]
    return tmp_image, tmp_path, tmp_bboxes


def rotate_image(image, mask, rotation, new_path):
    tmp_image = image.copy()
    tmp_mask = mask.copy()
    random_degree = float(randint(0, rotation - 1) % 360)
    tmp_image = ndimage.rotate(tmp_image, random_degree)
    tmp_mask = ndimage.rotate(tmp_mask, random_degree)
    bbox_abs = utils.get_bbox_by_mask(tmp_mask)

    xmin, ymin, xmax, ymax = bbox_abs.get_corners()
    fg_cut = tmp_image[ymin:ymax, xmin:xmax]
    mask_cut = tmp_mask[ymin:ymax, xmin:xmax]
    # set all mask values to 255 or 0
    for (x, y), value in np.ndenumerate(mask_cut):
        if mask_cut[x][y] > 200:
            mask_cut[x][y] = 255
        else:
            mask_cut[x][y] = 0
    h, w, _ = fg_cut.shape

    tmp_path = new_path[0:(len(new_path) - 4)] + "_r" + str(random_degree).replace('.0', '') + new_path[
                                                                                               (len(new_path) - 4):]

    return fg_cut, mask_cut, h, w, tmp_path


def blur_image(image, new_path, min_blur, max_blur):
    bf = randint(min_blur, max_blur)
    tmp_image = image.copy()
    tmp_image = cv2.blur(tmp_image, (bf, bf))
    tmp_path = new_path[0:(len(new_path) - 4)] + "_b" + str(bf).replace('.', '') + new_path[(len(new_path) - 4):]
    return tmp_image, tmp_path


def get_random_position_on_surface(w, h, bg_annotation_list, image_width, image_height):
    # choose random position inside a surface annotation
    l = len(bg_annotation_list)
    r = randint(0, l - 1)
    surface_box = bg_annotation_list[r].bbox
    surface_box = utils.norm_to_abs_bbox(surface_box, image_width, image_height)

    xmin, ymin, xmax, ymax = surface_box.get_corners()
    limit_left = xmin + w
    limit_right = xmax
    limit_down = ymax
    limit_up = max(h, ymin)

    if limit_right < limit_left or limit_up > limit_down:
        # print("Surface too small to place roi.")
        return None

    # choose random point in surface range (bottom of object)
    ymax  = randint(limit_up, limit_down)
    xmax = randint(limit_left, limit_right)

    # set other coordinates according to roi size
    xmin = xmax - w
    ymin = ymax - h

    bbox_rand = utils.BoundingBoxAbs((xmin+xmax)/2.0, (ymin+ymax)/2.0, xmax-xmin, ymax-ymin, image_width, image_height)

    return bbox_rand


def place_roi_on_bg(fg_cut, mask_cut, bg, bbox, new_path, bg_index):
    log = "place roi at: {}\n".format(bbox)
    # cut out object and background based on mask inside the roi
    try:
        mask_inv = cv2.bitwise_not(mask_cut)
        log = log + "fg_cut: {} - mask: {}\n".format(fg_cut.shape, mask_cut.shape)
        fg = cv2.bitwise_and(fg_cut, fg_cut, mask=mask_cut)
        xmin, ymin, xmax, ymax = bbox.get_corners()
        bg_cut = bg[ymin:ymax, xmin:xmax]
        log = log + "bg_cut: {} - mask: {}\n".format(bg_cut.shape, mask_inv.shape)
        bg_cut = cv2.bitwise_and(bg_cut, bg_cut, mask=mask_inv)
        roi = bg_cut + fg
        roi_h, roi_w, _ = roi.shape
        log = log + "roi size: {},{}\n".format(roi_w, roi_h)
        # insert roi in large image
        h, w, c = bg.shape
        new = np.zeros((h, w, c), np.uint8)
        new[0:h, 0:w] = bg
        new[ymin:ymin + roi.shape[0], xmin:xmin + roi.shape[1]] = roi
        tmp_path = new_path[0:(len(new_path) - 4)] + "_bg" + str(bg_index).replace('.', '') + new_path[
                                                                                              (len(new_path) - 4):]
        return new, tmp_path
    except:
        print("exception in place RoiOnBackground")
        print("log: \n{}".format(log))


def replace_annotation_in_image(image, mask, new_path, bg_list, num_bg, num_rotate, rotation_limit):
    results = []

    for i in range(num_rotate):
        fg_cut, mask_cut, h_cut, w_cut, rot_path = rotate_image(image, mask, rotation_limit, new_path)
        # check if mask and img have the same size, otherwise retry
        mh, mw = mask_cut.shape
        if not (mh == h_cut and mw == w_cut):
            print("fg_cut and mask_cut have different shapes! Skip.")
            continue

        for j in range(num_bg):
            bg_file = bg_list[randint(0, len(bg_list)-1)]
            print("choose bg: "+bg_file)
            bg_label_path = bg_file.replace("/images/", "/labels/").replace(".jpg", ".txt")
            bg = cv2.imread(bg_file, 1)
            bg_box_list = utils.read_annotations(bg_label_path)
            h, w, _ = bg.shape
            bbox_rand = get_random_position_on_surface(w_cut, h_cut, bg_box_list, w, h)
            if not bbox_rand:
                # todo: try again with smaller scaled object?
                print("couldn't place annotation on new background")
            else:
                bg_img, bg_path = place_roi_on_bg(fg_cut, mask_cut, bg, bbox_rand, rot_path, j)
                norm_box = utils.abs_to_norm_bbox(bbox_rand)
                result = (bg_img, bg_path, [norm_box])
                results.append(result)

    return results


def change_whole_image(image, path, annotation_list, norm_boxes, num_illuminate, num_scale, num_blur):
    # todo: norm box empty if no mask?
    for l in range(num_illuminate):
        l_img, l_path = change_illumination(image, path, 0.3, 0.7)
        for m in range(num_scale):
            s_img, s_path, s_boxes = change_scale(l_img, norm_boxes, l_path, 0.4, 1.0)
            for n in range(0, num_blur):
                b_img, b_path = blur_image(s_img, s_path, 1, 2)

                b_boxes = s_boxes
                if randint(0, 1) == 1:
                    b_img, b_path, b_boxes = mirror_image(b_img, s_boxes, b_path)

                # save
                save_image(b_img, annotation_list, b_boxes, b_path)


def save_image(image, annotation_list, bbox_list, image_file):
    # set new bounding boxes
    if not len(bbox_list) == len(annotation_list):
        print("ERROR: got {} annotations, but changed {} bounding boxes".format(len(bbox_list), len(annotation_list)))
        return
    for i in range(len(bbox_list)):
        annotation_list[i].bbox = bbox_list[i]

    # save annotations
    utils.save_annotations(image_file, annotation_list)

    # save image
    cv2.imwrite(image_file, image)
    # todo: option to generate rois


def multiply_dataset(input_dir, output_dir, bg_dir, num_rotate=1, num_illuminate=1, num_scale=1, num_blur=1, num_bg=1):

    print("input: " + input_dir)
    print("output: " + output_dir)

    bg_list = []
    use_bg = False
    if bg_dir is not None and not bg_dir == "" and num_bg > 0:
        print("try to read background data set")
        for dirname, dirnames, filenames in os.walk(bg_dir):
            for filename in filenames:
                file = dirname + '/' + filename
                if file.endswith(".jpg"):
                    l_file = file.replace("/images/", "/labels/").replace(".jpg", ".txt")
                    if not (os.path.isfile(l_file)):
                        print("error: Surface label file does not exist! Skipping image.")
                        continue
                    bg_list.append("{}".format(file))
        print("found {} backgrounds".format(len(bg_list)))
        if len(bg_list) > 0:
            use_bg = True
    else:
        print("background change disabled")

    print("multiply image (light, scale, blur, rotate): {}x{}x{}x{}x{} times".format(num_illuminate, num_scale,
                                                                                     num_blur, num_rotate, num_bg))
    rotation_limit = 80

    for dirname, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            file_path = dirname + '/' + filename

            if not os.path.exists(dirname.replace(input_dir, output_dir)):  # creates dir path
                os.makedirs(dirname.replace(input_dir, output_dir))
            roi_path = dirname.replace(input_dir, output_dir).replace("/images", "/rois").replace("/labels", "/rois")
            if not os.path.exists(roi_path):
                os.makedirs(roi_path)

            # deals with all None-Image files
            if imghdr.what(file_path) is None:

                if ".jpg" in filename or ".png" in filename:  # ignore empty images
                    continue

                if dirname.endswith("/labels"):  # ignores the label files, they will be written later
                    continue
                #if "train.txt" in filename or "test.txt" in filename:  # leaves the files empty to fill it later
                #    train_txt = open(file_path.replace(input_dir, output_dir).replace("test.txt", "train.txt"), 'a+')
                else:  # copy the files without changing
                    old_file = open(file_path, 'r')
                    new_file = open(file_path.replace(input_dir, output_dir), 'w+')
                    new_file.write(old_file.read())
                    old_file.close()  # close the streams
                    new_file.close()
                    continue

            if "mask." in file_path:  # ignore masks
                continue

            if "/rois" in file_path:  # ignore generated rois
                continue

            label_path = file_path.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
            mask_path = file_path.replace(".jpg", "_mask.jpg").replace(".png", "_mask.png")
            if not os.path.isfile(label_path):  # skip images with no labels
                continue

            has_mask = False
            if os.path.isfile(mask_path):   # check mask exists
                has_mask = True

            print(file_path)

            image = cv2.imread(file_path)
            mask = cv2.imread(mask_path, 0)
            annotation_list = utils.read_annotations(label_path)
            new_path = file_path.replace(input_dir, output_dir)

            results = []
            if has_mask and use_bg:
                results = replace_annotation_in_image(image, mask, new_path, bg_list, num_bg, num_rotate,
                                                      rotation_limit)

            bbox_list = []
            for a in annotation_list:
                bbox = copy.deepcopy(a.bbox)
                bbox_list.append(bbox)
            original = (image, new_path, bbox_list)
            results.append(original)
            for res in results:
                if not len(annotation_list) == len(res[2]):
                    print("ERROR: There are {} annotations, but only {} bounding boxes were changed. " +
                          "Skip.".format(len(annotation_list), len(res[2])))
                    continue

                change_whole_image(res[0], res[1], annotation_list, res[2], num_illuminate, num_scale, num_blur)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Data augmentation.py: Use this script to generate a new set of training data based on a smaller '
                    'annotated set or a set of images with masks images, defining object regions. The original data '
                    'be copied to a new directory and varied by scaling, blurring, illumination and if possible '
                    'rotation & background changing')

    parser.add_argument('imageset_dir', type=readable_dir, help='sys-path to the simple imageset')

    parser.add_argument('save_image_dir', type=valid_dir, help='sys-path to the folder where the result will be saved')

    parser.add_argument('-bg-dir', '--background_dir', type=str, default="", help='sys-path to the background images')

    parser.add_argument('-bg', '--background', type=int, default=0, help='number of background images to use (default=all)')
    parser.add_argument('-l', '--lighting', type=int, default=2, help='number of lighting changes (default=2)')
    parser.add_argument('-s', '--scale', type=int, default=2, help='number of scaling changes (default=2)')
    parser.add_argument('-b', '--blur', type=int, default=2, help='number of blurring the image (default=2)')
    parser.add_argument('-r', '--rotate', type=int, default=2, help='number of rotations of the image (default=2)')

    args = parser.parse_args()

    input_dir = args.imageset_dir
    output_dir = args.save_image_dir
    bg_dir = args.background_dir

    num_illuminate = args.lighting
    num_scale = args.scale
    num_blur = args.blur
    num_rotate = args.rotate
    num_bg = args.background

    multiply_dataset(input_dir, output_dir, bg_dir, num_rotate, num_illuminate, num_scale, num_blur, num_bg)

