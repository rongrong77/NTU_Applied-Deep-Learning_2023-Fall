import time
import cv2
import csv
import argparse
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from base.tool_patch import check_is_image, get_image_patch, image_padding
from base.metrics import get_metric

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=0, help="number of GPU")
parser.add_argument('--base_model_name', type=str, default='efficientnet-b0', help='base_model_name')
parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='pretrained encoder dataset')
parser.add_argument('--threshold', type=float, default=0.30, help='threshold for bgr mask')
parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='discriminator learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--image_test_dir', type=str, required=True, help='original image test dir')
parser.add_argument('--mask_test_dir', type=str, required=True, help='original mask test dir')
parser.add_argument('--focal_gamma', type=float, default=2, help='focal gamma')
opt = parser.parse_args()

device = torch.device("cuda:%s" % opt.gpu)

base_model_name = opt.base_model_name
lambda_bce = opt.lambda_bce
generator_lr = opt.generator_lr
threshold = opt.threshold
encoder_weights = opt.encoder_weights
focal_gamma = opt.focal_gamma

weight_folder = './Unet/stage2_dibco_' + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(threshold) + '_' + str(focal_gamma) + '/'
weight_list = sorted(os.listdir(weight_folder))
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list 
                    if weight_path.endswith('pth') and 'Unet' in weight_path]
print(weight_list)

models = []

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

weight_folder = './Unet/stage3_dibco_' + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(focal_gamma) + '/'
weight_list = os.listdir(weight_folder)
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list if 'unet_patch' in weight_path]
weight_list = sorted(weight_list)
print('stage3 weight', weight_list)

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()

weight_folder = './Unet/stage3_resize_dibco_' + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(focal_gamma) + '/'
weight_list = os.listdir(weight_folder)
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list if 'unet_global' in weight_path]
weight_list = sorted(weight_list)
print('stage3 resize weight:', weight_list)

model_normal_resize = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model_normal_resize.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model_normal_resize.to(device)
model_normal_resize.requires_grad_(False)
model_normal_resize.eval()

image_test_dir = opt.image_test_dir
mask_test_dir = opt.mask_test_dir
preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

value = int(256 * 0.5)
batch_size = 16
kernel = np.ones((7, 7), np.uint8)
resize_size = (512, 512)
skip_resize_ratio = 6
skip_max_length = 512
padding_resize_ratio = 4

save_root_dir = './Unet/predicted_image' + '_' + str(focal_gamma)
os.makedirs(save_root_dir, exist_ok=True)

save_fmeasure = {
    'stage3_normal':{
        '11':[[] for i in range(4)], 
        '13':[[] for i in range(4)], 
        '14':[[] for i in range(4)], 
        '16':[[] for i in range(4)],
        '17':[[] for i in range(4)],
        '18':[[] for i in range(4)],
        '19':[[] for i in range(4)]
        }
}

save_csv = open('./%s/metrics.csv' % save_root_dir, 'w')
save_csv_file = csv.writer(save_csv)
save_csv_file.writerow(['stage3_normal', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])

save_stage3_or_normal_dir = os.path.join(save_root_dir, 'stage3_normal')
os.makedirs(save_stage3_or_normal_dir, exist_ok=True)
for key in save_fmeasure['stage3_normal']:
    save_stage3_or_normal_dibco_dir = os.path.join(save_stage3_or_normal_dir, str(key))
    os.makedirs(save_stage3_or_normal_dibco_dir, exist_ok=True)

images = os.listdir(image_test_dir)
test_images = []
for image in images:
    if not check_is_image(image):
        print('not imgae', image)
        continue

    img_name = image.split('.')[0]
    gt_path_png = os.path.join(mask_test_dir, img_name + '.png')
    gt_path_bmp = os.path.join(mask_test_dir, img_name + '.bmp')
    if os.path.isfile(gt_path_png):
        gt_mask = gt_path_png
    elif os.path.isfile(gt_path_bmp):
        gt_mask = gt_path_bmp
    else:
        print(image, 'no mask')
        continue
    test_images.append( (os.path.join(image_test_dir, image), gt_mask) )

test_images = sorted(test_images, key=lambda x: (
    int(x[0].split('/')[-1].split('.')[0][5:9]),
    int(x[0].split('/')[-1].split('.')[0][10:])
    )
)

prev_dibco_year = test_images[0][0].split('/')[-1].split('.')[0][7:9]
total_time = time.time()
for test_image, test_mask in test_images:

    img_name = test_image.split('/')[-1].split('.')[0]
    image = cv2.imread(test_image)
    dibco_year = img_name.split('-')[0][7:]
    
    if prev_dibco_year != dibco_year:
        csv_tmp = ['average']
        for key in save_fmeasure:
            for sub_list in save_fmeasure[key][prev_dibco_year]:
                csv_tmp.append( sum(sub_list) / len(sub_list) )
            csv_tmp.extend([' ', ' '])
        save_csv_file.writerow(csv_tmp)
        save_csv_file.writerow([])
        prev_dibco_year = dibco_year

    gt_mask = cv2.imread(test_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask[gt_mask > 0] = 1

    print('processing the image:', img_name)
    h, w, _ = image.shape

    image_patches, poslist = get_image_patch(image, 256, 256, overlap=0.1, is_mask=False)
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
                preds.extend( torch.sigmoid(models[channel](target.to(device))).cpu() )
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

    image_patches, poslist = get_image_patch(merge_img, 256, 256, overlap=0.1, is_mask=False)

    color_patches = []
    for patch in image_patches:
        tmp = patch.astype(np.float32)
        color_patches.append(preprocess_input(tmp, input_space="BGR"))

    step = 0
    preds = []
    with torch.no_grad():
        while step < len(image_patches):
            ps = step
            pe = step + batch_size
            if pe >= len(image_patches):
                pe = len(image_patches)

            image_gray = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float().to(device)
            preds.extend( torch.sigmoid(model(image_gray).cpu()) )
            step += batch_size

    out_img = np.ones((h, w, 1)) * 255

    for i in range(len(image_patches)):
        patch = preds[i].permute(1, 2, 0).numpy() * 255

        start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
        h_cut = end_h - start_h
        w_cut = end_w - start_w

        tmp = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])
        out_img[start_h:end_h, start_w:end_w] = tmp

    stage3_out_img = out_img.astype(np.uint8)
    stage3_out_img[stage3_out_img > value] = 255
    stage3_out_img[stage3_out_img <= value] = 0
    stage3_out_img = np.squeeze(stage3_out_img, axis=-1)

    min_length = min(h, w)
    max_length = max(h, w)
    if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
        pass
    else:
        is_padded = False
        if min_length * padding_resize_ratio < max_length:
            image, position = image_padding(image)
            padded_size = image.shape[0]
            is_padded = True

        resized_img = cv2.resize(image, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
        resized_img = preprocess_input(resized_img, input_space="BGR")
        resized_img = np.expand_dims(resized_img, axis=0)
        resized_img = torch.from_numpy(resized_img).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            resized_mask_pred = model_normal_resize(resized_img)
            resized_mask_pred = torch.sigmoid(resized_mask_pred).cpu()
        
        resized_mask_pred = resized_mask_pred[0].permute(1, 2, 0).numpy() * 255
        resized_mask_pred = resized_mask_pred.astype(np.uint8)
        resized_mask_pred[resized_mask_pred > value] = 255
        resized_mask_pred[resized_mask_pred <= value] = 0

        if is_padded:
            resized_mask_pred = cv2.resize(resized_mask_pred, dsize=(padded_size, padded_size), interpolation=cv2.INTER_NEAREST)
            resized_mask_pred = resized_mask_pred[position[0]:position[1], position[2]:position[3]]
        else :
            resized_mask_pred = cv2.resize(resized_mask_pred, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
        resized_mask_pred = cv2.erode(resized_mask_pred, kernel, iterations=1)

    if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
        stage3_gray_img = np.copy(stage3_out_img)
        stage3_gray_img[stage3_gray_img > 0] = 1
        stage3_fmeasure, stage3_pfmeasure, stage3_psnr, stage3_drd = get_metric(stage3_gray_img, gt_mask)

        cv2.imwrite('%s/stage3_normal/%s/%s.png' % (save_root_dir, dibco_year, img_name), stage3_out_img)
        save_fmeasure['stage3_normal'][dibco_year][0].append(stage3_fmeasure)
        save_fmeasure['stage3_normal'][dibco_year][1].append(stage3_pfmeasure)
        save_fmeasure['stage3_normal'][dibco_year][2].append(stage3_psnr)
        save_fmeasure['stage3_normal'][dibco_year][3].append(stage3_drd)
        csv_tmp = [img_name, stage3_fmeasure, stage3_pfmeasure, stage3_psnr, stage3_drd]
    else:
        stage3_normal_or_img = np.bitwise_or(resized_mask_pred, stage3_out_img)
        stage3_normal_or_img_metric = np.copy(stage3_normal_or_img)
        stage3_normal_or_img_metric[stage3_normal_or_img_metric > 0] = 1
        stage3_normal_fmeasure, stage3_normal_pfmeasure, stage3_normal_psnr, stage3_normal_drd = get_metric(stage3_normal_or_img_metric, gt_mask)
        
        cv2.imwrite('%s/stage3_normal/%s/%s.png' % (save_root_dir, dibco_year, img_name), stage3_normal_or_img)
        save_fmeasure['stage3_normal'][dibco_year][0].append(stage3_normal_fmeasure)
        save_fmeasure['stage3_normal'][dibco_year][1].append(stage3_normal_pfmeasure)
        save_fmeasure['stage3_normal'][dibco_year][2].append(stage3_normal_psnr)
        save_fmeasure['stage3_normal'][dibco_year][3].append(stage3_normal_drd)
        csv_tmp = [img_name, stage3_normal_fmeasure, stage3_normal_pfmeasure, stage3_normal_psnr, stage3_normal_drd]

    save_csv_file.writerow(csv_tmp)

csv_tmp = ['average']
for key in save_fmeasure:
    for sub_list in save_fmeasure[key][dibco_year]:
        csv_tmp.append( sum(sub_list) / len(sub_list) )
    csv_tmp.extend([' ', ' '])
save_csv_file.writerow(csv_tmp)

save_csv.close()
