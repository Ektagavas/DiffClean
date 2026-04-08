## A pytorch reimplementation of [SSR-Net](https://www.ijcai.org/proceedings/2018/0150.pdf) by  

>_author_   :   oukohou  
>_time_     :   2019-09-26 16:44:48  
>_email_    :   oukohou@outlook.com

This code is forked from [here](https://github.com/oukohou/SSR_Net_Pytorch)


## Dataset
To finetune SSR-Net on your data, prepare train/val/test annotation files in the following format:

```shell
img_name,age,age_group
26_0_4_20170117195456708.jpg.chip.jpg,26,(20;29)
34_0_1_20170116002355232.jpg.chip.jpg,34,(30;39)
```
Replace the annotation file paths in SSR_NET_FT/train_SSR-Net.py and update the `base_path` to your dataset root.

## Fine-tuning
```shell
python train_SSR-Net.py
```