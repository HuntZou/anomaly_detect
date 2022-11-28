Anomaly detect project based on deep learning

该分支使用hrnet作为backbone，本意是想通过高分辨率获取更细节的信息，但无奈硬件设备不支持高分辨率train，所以实际效果并不好

ref: [Learning-Statistical-Texture-for-Semantic-Segmentation (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Learning_Statistical_Texture_for_Semantic_Segmentation_CVPR_2021_paper.pdf)

## Require

python -m pip install -r requirements.txt

## Dataset

[mvtec](ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz)

## Run

take class "bottle" for example:

```commandline
python train.py --class-name bottle
```