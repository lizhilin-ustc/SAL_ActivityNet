# SAL_ActivityNet

## Preparation
CUDA Version: 11.7

Pytorch: 1.12.0

Numpy: 1.23.1 

Python: 3.9.7

GPU: NVIDIA 3090

Dataset: the two-stream I3D features for ActivityNet1.3
## Training 
You can use the following command to train SAL:

```
python main_SA13_1.py
```

## Testing 

Our best model weights:

https://rec.ustc.edu.cn/share/a7ae72b0-4120-11f0-84c6-5fc3d25bb880
(code：nckc)

args.MODE = 'test' \
args.MODEL_FILE = '/path/to/model_best.pth.tar'

You can evaluate a trained model by running:
```
python main_SA13_1.py
```
