# label-gen

I'm trying to use pretrained model to generate annotations for instance segmentation. Steps below:
- First, use [mmdetection](https://github.com/open-mmlab/mmdetection) code to do inference, and generate json file in the process. The json file are made to be in [labelme](https://github.com/wkentaro/labelme) format, so that result can be easily visualized and checked. This step is done by `mmdet2labelme.py`
- Second, combine all labelme format json files into one single [COCO format](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) json file. After search I found [here](https://github.com/Tony607/labelme2coco) is good work by others. After some tweak about labelme imagedata and [numpy type](https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable) stuff, I have my version of `labelme2coco.py`
