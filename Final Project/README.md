# ADL 2023 Fall Final Project
## Environment
* Linux (Ubuntu)
* Python >= 3.6 (Pytorch)
* NVIDIA GPU + CUDA CuDNN
```
  cd 23-ADL-Final-Project
  pip install -r requirements.txt
```

## Dataset Download
* Use gdown to download dataset from Google Drive:
```
  bash download_dataset.sh
```

## Predict
### Example Trained Model Download
* Use gdown to download example trained model from Google Drive:
```
  bash download_model.sh
```
### Predict with Example Trained Model:
* We use the model trained with EfficientNet-B4 as the example:
```
  python evaluate.py --gpu 0 --base_model_name efficientnet-b4 --batch_size 16 --image_test_dir ./data/Testset/image/ --mask_test_dir ./data/Testset/mask/
```

## Train
* Before you run the following steps, make sure `cd 23-ADL-Final-Project`.
### Preprocess
```
  python preprocess/image_to_256.py
```
```
  python preprocess/image_to_512.py
```

### Enhancement
* Arguments:

| Key | Value | Help |
| :---: | :---: | :---: |
| --gpu | 0 | number of GPU |
| --epochs | 10 | number of epochs |
| --lambda_bce | 50.0 | bce weight |
| --base_model_name | efficientnet-b0 | base model name, i.e. efficientnet-b0 to efficientnet-b7 |
| --encoder_weights | imagenet | pretrained encoder dataset |
| --threshold | 0.30 | threshold for bgr mask |
| --generator_lr | 2e-4 | generator learning rate |
| --discriminator_lr | 2e-4 | discriminator learning rate |
| --batch_size | 16 | batch size, i.e. 16, 32, 64 |
| --image_train_dir | None | patched image train dir |
| --mask_train_dir | None | patched mask train dir |
| --image_test_dir | None | original image test dir |
| --mask_test_dir | None | original mask test dir |

* For example:
```
  python enhancement.py --gpu 0 --epochs 10 --base_model_name efficientnet-b0 --batch_size 16 --image_train_dir ./data/Trainset_256/image/ --mask_train_dir ./data/Trainset_256/mask/ --image_test_dir ./data/Testset/image/ --mask_test_dir ./data/Testset/mask/
```

### Local Prediciton
#### Local Prediction Preparation
* Arguments:

| Key | Value | Help |
| :---: | :---: | :---: |
| --gpu | 0 | number of GPU |
| --base_model_name | efficientnet-b0 | base model name, i.e. efficientnet-b0 to efficientnet-b7 |
| --lambda_bce | 50.0 | bce weight |
| --encoder_weights | imagenet | pretrained encoder dataset |
| --threshold | 0.30 | threshold for bgr mask |
| --generator_lr | 2e-4 | generator learning rate |
| --original_dir | None | original image train dir |
| --image_train_dir | None | patched image train dir |
| --mask_train_dir | None | patched mask train dir |
| --image_test_dir | None | original image test dir |
| --mask_test_dir | None | original mask test dir |

* For example:
```
  python to_local_prediction.py --gpu 0 --base_model_name efficientnet-b0 --original_dir data/Trainset/image/ --image_train_dir data/Trainset_256/image --mask_train_dir data/Trainset_256/mask --image_test_dir data/Testset/image --mask_test_dir data/Testset/mask
```
#### Local Prediction
* Arguments:

| Key | Value | Help |
| :---: | :---: | :---: |
| --gpu | 0 | number of GPU |
| --epochs | 10 | number of epochs |
| --lambda_bce | 50.0 | bce weight |
| --base_model_name | efficientnet-b0 | base model name, i.e. efficientnet-b0 to efficientnet-b7 |
| --encoder_weights | imagenet | pretrained encoder dataset |
| --threshold | 0.30 | threshold for bgr mask |
| --generator_lr | 2e-4 | generator learning rate |
| --discriminator_lr | 2e-4 | discriminator learning rate |
| --batch_size | 16 | batch size, i.e. 16, 32, 64 |
| --image_train_dir | None | patched enhanced image train dir |
| --mask_train_dir | None | patched enhanced mask train dir |
| --image_test_dir | None | enhanced image test dir |
| --mask_test_dir | None | original mask test dir |

* For example:
```
  python local_prediction.py --gpu 0 --epochs 10 --base_model_name efficientnet-b0 --batch_size 16 --image_train_dir ./Unet/image_for_local_prediciton/train/patch/image/ --mask_train_dir ./Unet/image_for_local_prediciton/train/patch/mask/ --image_test_dir ./Unet/image_for_local_prediciton/test/ --mask_test_dir ./data/Testset/mask/
```

### Gobal Prediction
* Arguments:

| Key | Value | Help |
| :---: | :---: | :---: |
| --gpu | 0 | number of GPU |
| --epochs | 150 | number of epochs |
| --lambda_bce | 50.0 | bce weight |
| --base_model_name | efficientnet-b0 | base model name, i.e. efficientnet-b0 to efficientnet-b7 |
| --encoder_weights | imagenet | pretrained encoder dataset |
| --threshold | 0.30 | threshold for bgr mask |
| --generator_lr | 2e-4 | generator learning rate |
| --discriminator_lr | 2e-4 | discriminator learning rate |
| --batch_size | 16 | batch size, i.e. 16, 32, 64 |
| --image_train_dir | None | 512 resized image train dir |
| --mask_train_dir | None | 512 resized mask train dir |
| --image_test_dir | None | original image test dir |
| --mask_test_dir | None | original mask test dir |

* For example:
```
  python gobal_prediction.py --gpu 0 --epochs 150 --base_model_name efficientnet-b0 --batch_size 4 --image_train_dir ./data/Trainset_512/image/ --mask_train_dir ./data/Trainset_512/mask/ --image_test_dir ./data/Testset/image/ --mask_test_dir ./data/Testset/mask/
```

### Evaluate
* Arguments:

| Key | Value | Help |
| :---: | :---: | :---: |
| --gpu | 0 | number of GPU |
| --base_model_name | efficientnet-b0 | base model name, i.e. efficientnet-b0 to efficientnet-b7 |
| --lambda_bce | 50.0 | bce weight |
| --encoder_weights | imagenet | pretrained encoder dataset |
| --threshold | 0.30 | threshold for bgr mask |
| --generator_lr | 2e-4 | generator learning rate |
| --discriminator_lr | 2e-4 | discriminator learning rate |
| --batch_size | 16 | batch size, i.e. 16, 32, 64 |
| --image_test_dir | None | original image test dir |
| --mask_test_dir | None | original mask test dir |

* For example:
```
  python evaluate.py --gpu 0 --base_model_name efficientnet-b0 --batch_size 16 --image_test_dir ./data/Testset/image/ --mask_test_dir ./data/Testset/mask/
```
