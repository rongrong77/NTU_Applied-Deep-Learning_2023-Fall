import os
import numpy as np
import torch
import cv2
import argparse
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from base.tool_patch import get_image_patch_deep, get_image_patch, check_is_image

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=0, help="number of GPU")
parser.add_argument('--base_model_name', type=str, default='efficientnet-b0', help='base model name')
parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='pretrained encoder dataset')
parser.add_argument('--threshold', type=float, default=0.30, help='threshold for bgr mask')
parser.add_argument('--focal_gamma', type=float, default=2, help='focal gamma')
parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
parser.add_argument('--original_dir', type=str, required=True, help='original image train dir')
parser.add_argument('--image_train_dir', type=str, required=True, help='patched image train dir')
parser.add_argument('--mask_train_dir', type=str, required=True, help='patched mask train dir')
parser.add_argument('--image_test_dir', type=str, required=True, help='original image test dir')
parser.add_argument('--mask_test_dir', type=str, required=True, help='original mask test dir')
opt = parser.parse_args()

device = torch.device("cuda:%s" % opt.gpu)

models = []
base_model_name = opt.base_model_name
lambda_bce = opt.lambda_bce
generator_lr = opt.generator_lr
threshold = opt.threshold
encoder_weights = opt.encoder_weights

weight_folder = './Unet/stage2_dibco_' + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(threshold) + '_' + str(opt.focal_gamma) + '/'
weight_list = sorted(os.listdir(weight_folder))
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list 
                    if weight_path.endswith('pth') and 'Unet' in weight_path]

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[1], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[2], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[3], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

batch_size = 12
preprocess_input = get_preprocessing_fn(base_model_name, pretrained='imagenet')

image_save_path = './Unet/image_for_local_prediciton' + '_' + str(opt.focal_gamma)
os.makedirs(image_save_path, exist_ok=True)

train_image_save_path = os.path.join(image_save_path, 'train')
os.makedirs(train_image_save_path, exist_ok=True)

test_image_save_path = os.path.join(image_save_path, 'test')
os.makedirs(test_image_save_path, exist_ok=True)

# patch directory
patch_save_path = os.path.join(train_image_save_path, 'patch')
os.makedirs(patch_save_path, exist_ok=True)

patch_train_image_save_path = os.path.join(patch_save_path, 'image')
os.makedirs(patch_train_image_save_path, exist_ok=True)

patch_train_mask_save_path = os.path.join(patch_save_path, 'mask')
os.makedirs(patch_train_mask_save_path, exist_ok=True)

stage3_overlap_ratio = 0.3
scale_list = [0.75, 1.00, 1.25, 1.50]
rotation = [0, 3]
reshape = (256, 256)

predict_overlap_ratio = 0.1
crop_h = 256
crop_w = 256
'''
image_train_dir = opt.image_train_dir
mask_train_dir = opt.mask_train_dir
images = os.listdir(image_train_dir)
for img in images:
    if not check_is_image(img):
        print('not image', img)
        continue

    image = cv2.imread(os.path.join(image_train_dir, img))

    image_name = img.split('.')[0]
    print('processing the image:', img)

    # find and read mask file
    if os.path.isfile(os.path.join(mask_train_dir, image_name + '.png')):
        mask = cv2.imread(os.path.join(mask_train_dir, image_name + '.png'), cv2.IMREAD_GRAYSCALE)
    elif os.path.isfile(os.path.join(mask_train_dir, image_name + '.bmp')):
        mask = cv2.imread(os.path.join(mask_train_dir, image_name + '.bmp'), cv2.IMREAD_GRAYSCALE)
    else:
        print(img, 'no mask')
        exit(1)

    mask[mask < 190] = 0
    mask[mask >= 190] = 255
    
    h, w, _ = image.shape
    image_patches, poslist = get_image_patch(image, crop_h, crop_w, overlap=predict_overlap_ratio, is_mask=False)

    merge_img = np.ones((h, w, 3))
    out_imgs = []

    for channel in range(4):
        color_patches = []
        for patch in image_patches:
            tmp = patch.astype(np.float32)
            if channel != 3:
                color_patches.append(preprocess_input(tmp[:, :, channel:channel+1]))
            else:
                color_patches.append(preprocess_input(np.expand_dims( cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY), axis=-1 )))

        step = 0
        preds = []
        with torch.no_grad():
            while step < len(image_patches):
                ps = step
                pe = step + batch_size
                if pe >= len(image_patches):
                    pe = len(image_patches)

                target = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float()
                preds.extend(torch.sigmoid(models[channel](target.to(device))).cpu())
                step += batch_size

        out_img = np.ones((h, w, 1)) * 255
        for i in range(len(image_patches)):
            patch = preds[i].permute(1, 2, 0).numpy() * 255

            start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
            h_cut = end_h - start_h
            w_cut = end_w - start_w

            tmp = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])
            out_img[start_h:end_h, start_w:end_w] = tmp
        out_imgs.append(out_img)

    merge_img[:, :, 0:1] = (out_imgs[0] + out_imgs[3]) / 2.
    merge_img[:, :, 1:2] = (out_imgs[1] + out_imgs[3]) / 2.
    merge_img[:, :, 2:3] = (out_imgs[2] + out_imgs[3]) / 2.
    merge_img = merge_img.astype(np.uint8)
    cv2.imwrite('%s/%s.png' % (train_image_save_path, image_name), merge_img)

    scale_cnt = 0
    for scale in scale_list:
        crpW = int(scale * crop_w)
        crpH = int(scale * crop_h)

        image_patches, poslist = get_image_patch_deep(merge_img, crpH, crpW, reshape, overlap=stage3_overlap_ratio)
        mask_patches, poslist = get_image_patch_deep(mask, crpH, crpW, reshape, overlap=stage3_overlap_ratio)

        for idx in range(len(image_patches)):
            image_patch = image_patches[idx]
            mask_patch = mask_patches[idx]

            for k in rotation:
                img_tmp = np.rot90(image_patch, k)
                mask_tmp = np.rot90(mask_patch, k)
                cv2.imwrite('%s/%s_s%dr%di%d.png' % (patch_train_image_save_path, image_name, scale_cnt, k, idx), img_tmp)
                cv2.imwrite('%s/%s_s%dr%di%d.png' % (patch_train_mask_save_path, image_name, scale_cnt, k, idx), mask_tmp)
        scale_cnt += 1
'''
image_test_dir = opt.image_test_dir
images = os.listdir(image_test_dir)
for img in images:
    if not check_is_image(img):
        print('not image', img)
        continue

    image = cv2.imread(os.path.join(image_test_dir, img))
    
    image_name = img.split('.')[0]
    print('processing the image:', img)

    h, w, _ = image.shape
    image_patches, poslist = get_image_patch(image, crop_h, crop_w, overlap=predict_overlap_ratio, is_mask=False)

    merge_img = np.ones((h, w, 3))
    out_imgs = []

    for channel in range(4):
        color_patches = []
        for patch in image_patches:
            tmp = patch.astype(np.float32)
            if channel != 3:
                color_patches.append(preprocess_input(tmp[:, :, channel:channel+1]))
            else:
                color_patches.append(preprocess_input(np.expand_dims( cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY), axis=-1 )))

        step = 0
        preds = []
        with torch.no_grad():
            while step < len(image_patches):
                ps = step
                pe = step + batch_size
                if pe >= len(image_patches):
                    pe = len(image_patches)

                target = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float()
                preds.extend(torch.sigmoid(models[channel](target.to(device))).cpu())
                step += batch_size

        out_img = np.ones((h, w, 1)) * 255
        for i in range(len(image_patches)):
            patch = preds[i].permute(1, 2, 0).numpy() * 255

            start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
            h_cut = end_h - start_h
            w_cut = end_w - start_w

            tmp = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])
            out_img[start_h:end_h, start_w:end_w] = tmp
        out_imgs.append(out_img)

    merge_img[:, :, 0:1] = (out_imgs[0] + out_imgs[3]) / 2.
    merge_img[:, :, 1:2] = (out_imgs[1] + out_imgs[3]) / 2.
    merge_img[:, :, 2:3] = (out_imgs[2] + out_imgs[3]) / 2.
    merge_img = merge_img.astype(np.uint8)
    cv2.imwrite('%s/%s.png' % (test_image_save_path, image_name), merge_img)
