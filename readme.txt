1、labelling
2、yolo requirement ###
3、data augumentation
4、tranfer data format 2 txt
	pip install labelme2yolo
	运行labelme2yolo --json_dir E:\   --val_size 0.3
	*yaml，path
5、train.py:(1)yaml；(2)workers，
	：pip install numpy==1.26.4
	os.environ["GIT_PYTHON_REFRESH"] = "quiet"
6、val.py "--data" *.yaml;"--weights" best.py
7. detect.py "--weights" best.py ;"source" 
8、export.py
    '--data' *.yaml;"--weights" best.py; default=onnx
	pip install onnx==1.13.0 protobuf==3.20.3 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
	pip install onnxruntime==1.15.1 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
9. val.py   "--weights" best.onnx

nvidia-smi


conda env remove --name test
conda create --name tester --clone yolo

1. 下载权重文件
2. 下载字体 Arial.ttf
3. 运行项目报错
    w, h = self.font.getsize(label)  # text width, height
AttributeError: 'FreeTypeFont' object has no attribute 'getsize'

D:\projects\yolov5-7.0>pip install pillow==9.1.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


4. 参数修改
workers 、

5.警告
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.


6. Numpy兼容问题
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

pip install numpy==1.26.4


7. 运行项目报错
The git executable must be specified in one of the following ways:
    - be included in your $PATH
    - be set via $GIT_PYTHON_GIT_EXECUTABLE
    - explicitly set via git.refresh(<full-path-to-git-executable>)

All git commands will error until this is rectified.

This initial message can be silenced or aggravated in the future by setting the
$GIT_PYTHON_REFRESH environment variable. Use one of the following values:
    - quiet|q|silence|s|silent|none|n|0: for no message or exception
    - warn|w|warning|log|l|1: for a warning message (logging level CRITICAL, displayed by default)
    - error|e|exception|raise|r|2: for a raised exception

Example:
    export GIT_PYTHON_REFRESH=quiet


解决
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

8. miniconda安装
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu

9. 环境报编码错误
UnicodeDecodeError: 'gbk' codec can't decode byte 0x98 in position 1048: illegal multibyte sequence

set PYTHONUTF8=1
$env:PYTHONUTF8=1

10.划分数据集脚本

https://github.com/rooneysh/Labelme2YOLO
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2


pip install labelme2yolo
https://pypi.org/project/labelme2yolo/
labelme2yolo --json_dir D:\projects\augment\augmented --val_size 0.2


11. miniconda下载地址
https://repo.anaconda.com/miniconda/
