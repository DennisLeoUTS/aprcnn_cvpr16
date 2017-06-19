Weakly-supervised Dataset augmented Part-based RCNNs for Fine-grained Category Detection
===============
This work is created by Zhe Xu from UTS. Credit to Ning Zhang, Jeff Donahue, Ross Girshick and Trevor Darrell from UC Berkeley who share the original RCNN and Part-based RCNN code.


### Prerequisites
0. **Caffe**
 - Download caffe from http://caffe.berkeleyvision.org/ and follow the instructions to install. 
 - Change caffe matlab wrapper path in init.m

0. **RCNN**
  - A modified RCNN code is attached in the code patch. 
  - If you want to access the original RCNN code to see the modifications, download source code from https://github.com/rbgirshick/rcnn and follow the instructions to install.
  - Follow rcnn instructions to train the part detectors.

0. **Liblinear**
  - Download liblinear package from http://www.csie.ntu.edu.tw/~cjlin/liblinear/  
  - Download MALSAR package from http://www.public.asu.edu/~jye02/Software/MALSAR/  

Annotation/ has annotated part boxes on CUB200-2011 dataset.  

### Cached Files  
  - Recommended file: https://drive.google.com/file/d/0Bwo0SFiZwl3JR1RFdnM0R3EyNDA/view  
  - Full file: https://drive.google.com/file/d/0Bwo0SFiZwl3JSmZNXzZJVG5FbEE/view  

### Usage  
  - If you have installed cached files, run run_prcnn_sample.m to get results  
  - To run the whole procedure of extracting selective search data -> finetune part cnns -> train RCNN part detectors -> get augmented   training data from weak dataset -> re-finetune part cnns -> get final classification results, go to ./rcnn directory and run rcnn_run.m.  
  - Modify the paths in config_prcnn.m and get_bird_data.m  

###Bug report  
If you have any issues running the codes, please contact Zhe Xu (xz303010@gmail.com).