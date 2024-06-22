# mmlab-pose
## Introduction
This repository serves as tools for using the ap10k model and others from the [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

Here, all you will need is the weights and configs downloaded from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)


## Setup
Make sure to do the following, as the packages are required to run the keypoints script:
```bash
cd mmcv
pip install -e .

cd ..
cd ViTPose
pip install -e .

```

This installs the edited mmpose package and the ViTPose package needed to use the ViTPose models.

### Splitting the model
Certain models are grouped with several models being contained in one .pth file, ('+' in the .pth file) you will need to split the model in its parts.
To split the model, you will need to run the following command:
```bash
python ViTPose/tools/split_model.py --source /path/to/model.pth
```
