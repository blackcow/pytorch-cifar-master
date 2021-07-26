import cv2
import numpy as np

# from jpeg2dct.numpy import load, loads
def upscale(img, scale=2):
    h, w, _ = img.shape
    new_h, new_w = h * scale, w * scale
    return cv2.resize(img, (new_h, new_w))


# def convert_to_dct(img_location):
#     dct_y, dct_cb, dct_cr = load(img_location)
#     with open(img_location, 'rb') as src:
#         buffer = src.read()
#     dct_y, dct_cb, dct_cr = loads(buffer)
#     return [dct_y,dct_cb,dct_cr]

def center_crop(img_location, dim):
    ''' From https://medium.com/curious-manava/center-crop-and-scaling-in-opencv-using-python-279c1bb77c74'''
    img = cv2.imread(img_location)
    # print(img_location)
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    # tmp_location = '/data/niuzh/Code/Fair-AT/datset/tmp_img.jpg'
    # cv2.imwrite(tmp_location, crop_img[:, :, [2, 1, 0]])

    # return tmp_location, crop_img
    return crop_img


def normalize_channel(c):
    mean = np.mean(c)
    std = np.std(c)
    c = (c - 255.0 * np.array(mean)) / np.array(std)
    return c


def normalize(img):
    h, w, c = img
    return (normalize_channel(h), normalize_channel(w), normalize_channel(c))
