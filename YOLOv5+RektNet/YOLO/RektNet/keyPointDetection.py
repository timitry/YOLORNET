import torch
import cv2
import numpy as np
import argparse
import sys
import os
import sys
import shutil
import RektNet.kpdetUtils
from RektNet.kpdetUtils import vis_tensor_and_save, prep_image

from RektNet.keypoint_net import KeypointNet



#Процедура отрисовки точек на исходном кадре, возвращает пересчитанный лист с 7 точками для текущего конуса

def vis_tensor_and_save_result(image, h, w, tensor_output, image_name, output_uri, x ,y):
    vis_tmp_path = "/tmp/detect/"  # !!!don't specify this path outside of /tmp/, otherwise important files could be removed!!!
    vis_path = "/outputs/visualization/"
    a = list()
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (127, 255, 127), (255, 127, 127)]
    i = 0
    for pt in np.array(tensor_output):
        cv2.circle(image, (y + int(pt[0] * w), int(x + pt[1] * h)), 3, colors[i], -1)
        a.append((y + int(pt[0] * w), int(x + pt[1] * h)))
        i += 1
    if not cv2.imwrite(os.path.join(vis_tmp_path, image_name + "_result.jpg"), image):
        raise Exception("Could not write image")  # opencv wongi't give you error for incorrect image but return False instead, so we have to do it manually

    os.rename(os.path.join(vis_tmp_path, image_name + "_result.jpg"),
              os.path.join(output_uri, image_name + "_result.jpg"))
    return a


#Детект прогоняет кроп через сетку, возвращает тензор точек в относительных координатах


def detect(model,img,img_size,output,flip,rotate,iteration):

    output_path = output

    model_path = model

    model_filepath = model_path

    #image_path = img

    #image_filepath = image_path

    #img_name = '_'.join(image_filepath.split('/')[-1].split('.')[0].split('_')[-5:])
    img_name = str(iteration)
    image_size = (img_size, img_size)

    image = img.numpy()
    h, w, _ = image.shape

    image = prep_image(image=image,target_image_size=image_size)
    image = (image.transpose((2, 0, 1)) / 255.0)[np.newaxis, :]
    image = torch.from_numpy(image).type('torch.FloatTensor')

    model = KeypointNet()
    model.load_state_dict(torch.load(model_filepath).get('model'))
    model.eval()
    output = model(image)
    out = np.empty(shape=(0, output[0][0].shape[2]))
    for o in output[0][0]:
        chan = np.array(o.cpu().data)
        cmin = chan.min()
        cmax = chan.max()
        chan -= cmin
        chan /= cmax - cmin
        out = np.concatenate((out, chan), axis=0)
    #cv2.imwrite(output_path + img_name + "_hm.jpg", out * 255)
    #print(f'please check the output image here: {output_path + img_name + "_hm.jpg", out * 255}')
    print("Dots tensor for " + str(iteration) + ' cone: \n', output[1][0].cpu().data)


    #image = cv2.imread(image_filepath)
    image = img.numpy()
    h, w, _ = image.shape

    vis_tensor_and_save(image=image, h=h, w=w, tensor_output=output[1][0].cpu().data, image_name=img_name,
                        output_uri=output_path)
    return output[1][0].cpu().data


#Кроппер по изображению и файлу с информацией о ббоксах кропает конусы, запускает для конкретного кропа детект и отрисовку возвращенных координат, в конце возвращает лист с точками в абсолютных координатах.

def cropper(image_path, bbFile_path, outpath, weights):
    f = open(bbFile_path, 'r', encoding='utf-8')
    img_name = '_'.join(image_path.split('/')[-1].split('.')[0].split('_')[-5:])
    image = cv2.imread(image_path)
    tmp = image
    assert not isinstance(image, type(None)), 'image not found'
    h, w = image.shape[:2]
    image = torch.from_numpy(image).type('torch.FloatTensor')
    result = list()
    i = 1
    for line in f:
        a = list(map(float, line.split()))
        x1 = int(a[2] * h - a[4] * h * 0.5)
        x2 = int(a[2] * h + a[4] * h * 0.5)
        y1 = int(a[1] * w - a[3] * w * 0.5)
        y2 = int(a[1] * w + a[3] * w * 0.5)
        img = image[x1:x2, y1:y2]
        wc = y2 - y1
        hc = x2 - x1

        a = detect(model=weights, img=img, img_size=80, output=outpath, flip=False, rotate=False, iteration=i) #раньше было args.model для автономной работы
        b = vis_tensor_and_save_result(tmp, hc, wc, a, img_name, outpath, x1, y1)
        result.append(b)
#
        i += 1
    cv2.imwrite(image_path[:-4] + "_result.jpg", tmp)
    f.close()

    return result






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keypoints Visualization')
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})

    parser.add_argument('--model', help='path to model', type=str, required=True)
    parser.add_argument('--img', help='path to single image', type=str, default="gs://mit-dut-driverless-external/ConeColourLabels/vid_3_frame_22063_0.jpg")
    parser.add_argument('--img_size', help='image size', default=80, type=int)
    parser.add_argument('--output', help='path to upload the detection', default="outputs/visualization/")
    parser.add_argument('--bbdata', type=str) #текстовый файл с разметкой
    add_bool_arg('flip', default=False, help='flip image')
    add_bool_arg('rotate', default=False, help='rotate image')

    args = parser.parse_args(sys.argv[1:])

    #res = cropper(args.img, args.bbdata)
    #print(res)

