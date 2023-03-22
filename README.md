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
# this commaned train all 15 classes, you can change it in config.py
python -u src/run/train.py
```

or you can just export test dataset result or generate pseudo anom by run `src/run/[export_img.py|pseudo_anom.py]`

## Pseudo anom

![bottle](./images/pseudo_anom/bottle.png)
![capsule](./images/pseudo_anom/capsule.png)
![carpet](./images/pseudo_anom/carpet.png)
![pill](./images/pseudo_anom/pill.png)

## Result

TODO

## troubleshooting

You can change any running configuration in config.py to satisfy your env.

If error occurred when training

```commandline
AttributeError: Can't pickle local object 'ADMvTec.__init__.<locals>.<lambda>'
```
Maybe you are on Windows, since it doesn't support pickle lambda expression, you should change num_work to 0 in config.py