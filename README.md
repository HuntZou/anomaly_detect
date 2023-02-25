Anomaly detect project based on deep learning

ref: [Learning-Statistical-Texture-for-Semantic-Segmentation (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Learning_Statistical_Texture_for_Semantic_Segmentation_CVPR_2021_paper.pdf)

## Require

python -m pip install -r requirements.txt

## Dataset

[mvtec](ftp://guest:GU%2E205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz)

## Run

modify config file `config.py` to fit your env first, like dataset_path is the dataset dir where `mvtec_anomaly_detection.tar.xz` located

take class "bottle" for example:

```commandline
python -u train.py --class-name bottle
```

## fine-tuning for special class

capsule: src/datasets/online_supervisor.py:206&288 be commended