
## Prerequisites

- PyTorch >= 1.1
- Python == 3.6
- gcc 5

## Getting Started

### Installation

Clone the github repository. We will call the cloned directory as `$DBG_ROOT`.  
```bash
cd $DBG_ROOT
```
Firstly, you should compile our proposal feature generation layers. 

Compile **pytorch-version** proposal feature generation layers:
```bash
cd pytorch/custom_op
python setup.py install
```

### Download Datasets

Prepare ActivityNet 1.3 dataset. You can use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube. Some videos have been deleted from YouTube，and you can also ask for the whole dataset by email.

Extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.


For generating the video features, scripts in `./tools` will help you to start from scrach.

### Training
We provide training code of pytorch version. Please check the `feat_dir` in `config/config.yaml` and follow these steps to train your model: 
#### 1. Training
```bash
python pytorch/train.py config/config.yaml
```
#### 2. Testing
```bash
python pytorch/test.py config/config.yaml
```

#### 3. Postprocessing
```bash
python post_processing.py output/result/ results/result_proposals.json
```

#### 4. Evaluation
```bash
python eval.py results/result_proposals.json
```

### Testing
```bash
python pytorch/test.py config/config_pretrained.yaml
python post_processing.py output/result/ results/result_proposals.json
python eval.py results/result_proposals.json
```
