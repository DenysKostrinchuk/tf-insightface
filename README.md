# tf-InsightFace

## Accuracy testing
### Calulate accuracy
1) Download dataset [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71peklHb0pGdDl6R28) and uzip all images in one folder;
2) Put pb weights to pretrained folder.
2) You need to transform each record in images_name_1d.npy according to where your folder with the dataset is located.
From ```['/home/jetson/denis/img_celeba/000001.jpg', '2880']``` to ```['<your_path>/img_celeba/000001.jpg', '2880']``` and put it to folder ```deploy/```;
3) Run script
```
 cd tf-insightface
 python3 apps/run_dataset.py
```
### Parameters accuracy
```
--dataset: Path to file with dataset records (default = 'images_name_1d.npy');
--threshold: Threshold for comparing distance (default=1.0);
--step_accuracy: How often should save accuracy, in iterations (default=50);
--save_one_in: How often save "trt_1d_result.json" file with accuracy, in iterations (default=1000)
```
### Accuracy results
Each <save_one_in> iteraion file "tf_1d_result.json" will be stored. This file consist num_iteration and accuracy on this iteration.


## References

* [Deng, Jiankang, Jia Guo, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018).](https://arxiv.org/abs/1801.07698)

* Official Implementation (mxnet): [deepinsight/insightface](https://github.com/deepinsight/insightface)

* Third Party Implementation (tensorflow): [auroua/InsightFace_TF](https://github.com/auroua/InsightFace_TF)