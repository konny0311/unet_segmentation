import os
import glob
import json
import cv2
import numpy as np
import pprint
import colorsys
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from tqdm import tqdm
from enum import Enum

BASE_DIR = os.path.join('.', 'images')
IMAGE_DIR = os.path.join(BASE_DIR, 'original/')
ANNOTAION_DIR = os.path.join(BASE_DIR, 'annot/')
MASK_DIR = os.path.join(BASE_DIR, 'mask_masks/')


REGION_COLOR = (0, 1, 0)
REGION_TH = 10 #90

class DrawType(Enum):
    POLYGON = 0
    LINE = 1
    CIRCLE = 2

CLASS_LIST = ["dog"]


def main():
    if not os.path.exists(MASK_DIR):
        print('not exists path : ', MASK_DIR)
        os.mkdir(MASK_DIR)
        print('   -> maked')

    json_files = os.listdir(ANNOTAION_DIR)
    print('')
    print('target count : ', len(json_files))
    print('')

    for i,json_file in enumerate(json_files):
        if json_file.endswith('.json'):
            k = i + 1
            print('###### [%03d] : ' % k, ANNOTAION_DIR + json_file)
            name = json_file.replace('.json', '')
            out_file = name + '.png'
            create_annotation_img(ANNOTAION_DIR + json_file, IMAGE_DIR + out_file, MASK_DIR + out_file)

# def create_npy(target_dir):
#     inp_img_path = os.path.join(IMAGE_DIR, target_dir)
#     print(inp_img_path)
#     inp_anno_path = os.path.join(ANNOTAION_DIR, target_dir)
#     print(inp_anno_path)
#     out_mask_path = os.path.join(MASK_DIR, target_dir)
#     out_overlay_path = os.path.join(OVERLAY_DIR, target_dir)
#     out_seg_path = os.path.join(SEGMENTATION_DIR, target_dir)
#     if not os.path.exists(inp_img_path):
#         print('not exists related image dir. : ', inp_img_path)
#         return False

#     if not os.path.exists(out_mask_path):
#         os.mkdir(out_mask_path)
#     if not os.path.exists(out_overlay_path):
#         os.mkdir(out_overlay_path)
#     if not os.path.exists(out_seg_path):
#         os.mkdir(out_seg_path)

#     files = glob.glob(os.path.join(inp_anno_path, '*.json'))
#     files.sort()

#     print('')
#     print('### target annotation file : ', len(files))
#     print('')

#     pbar = tqdm(total=len(files), desc="Create", unit=" Files")
#     for _, file in enumerate(files):
#         create_annotation_img(file, inp_img_path, out_mask_path, out_overlay_path, out_seg_path)
#         pbar.update(1)
#     pbar.close()
#     return True


def create_annotation_img(anno_json, inp_img_path, out_mask_path):
    print(out_mask_path)
    jf = json.load(open(anno_json))
    image_name_base, _ = os.path.splitext(os.path.basename(anno_json))

    # original_image_path = os.path.join(inp_img_path, image_name_base)
    # print(original_image_path)
    org_img = cv2.imread(inp_img_path)
    # if org_img is None:
    #     org_img = cv2.imread(original_image_path + '.jpeg')
    org_img = org_img.astype(np.uint8)
    img_h, img_w, img_c = np.shape(org_img)

    seg_img = None
    reg_colors = random_colors(len(CLASS_LIST))
    img = np.zeros(np.shape(org_img), dtype=np.uint8)
    for k, shape in enumerate(jf['shapes']):
        contours = shape['points']
        if not shape['label'] in CLASS_LIST:
            continue

        label_idx = 0
        one_hot_vec = np.repeat([0], 1 + len(CLASS_LIST))
        for k, c in enumerate(CLASS_LIST):
            if shape['label'] == c:
                one_hot_vec[k + 1] = 1
                label_idx = k
                break

        mask = contours
        img_g = create_mask_image(np.shape(org_img), mask, draw_type=DrawType.POLYGON)
        #img = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)
        #img = np.zeros(np.shape(org_img), dtype=np.uint8)
        if img is None:
            continue
        img[img_g > 100] = reg_colors[label_idx]

        file_name = '%s_%d.png' % (image_name_base, k)
        # file_path = os.path.join(out_mask_path, file_name)
        cv2.imwrite(out_mask_path, img)



def resize_mask(ratio_h, ratio_w, mask):
    new_mask = np.array(mask, dtype='float32')
    new_mask[:, 0] *= ratio_w
    new_mask[:, 1] *= ratio_h
    new_mask = new_mask.astype('int32')
    return new_mask


def create_region_image(image_shape, mask, reg_color=REGION_COLOR):
    img_src = np.zeros((image_shape[0], image_shape[1], 3))
    x, y, w, h = cv2.boundingRect(mask)
    img = cv2.rectangle(img_src, (x, y), (x+w, y+h), reg_color, 4)
    #img = img * 255
    region_size = w * h
    if region_size < REGION_TH:
        return None, None
    return img, (x, y, x+w, y+h)


def create_mask_image(image_shape, mask, draw_type=DrawType.POLYGON):
    img_src = np.zeros(image_shape[:2])
    img = Image.fromarray(img_src)
    xy = list(map(tuple, mask))
    if draw_type == DrawType.POLYGON:
        ImageDraw.Draw(img).polygon(xy=xy, outline=255, fill=255)
    elif draw_type == DrawType.LINE:
        ImageDraw.Draw(img).line(xy=xy, width=3, fill=255)
    elif draw_type == DrawType.CIRCLE:
        if len(xy) > 0:
            c_x, c_y = xy[0]
            radius = 4
            xy = [(c_x - radius, c_y - radius), (c_x + radius, c_y + radius)]
            ImageDraw.Draw(img).ellipse(xy, outline=255, fill=255)
    img = np.array(img) * 255
    return img


def random_colors(N):
    rgb_colors = []
    for i in range(N):
        hsv = i/N, 0.8, 1.0
        rgb = colorsys.hsv_to_rgb(*hsv)
        rgb = tuple((int(val * 255) for val in rgb))
        rgb_colors.append(rgb)
    return rgb_colors


if __name__ == '__main__':
    main()
