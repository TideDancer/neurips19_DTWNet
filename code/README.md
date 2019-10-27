------- environment --------

1. install required package

pip install Cython

pip install cffi

pip install pytorch

pip install torchvision

pip install visdom

pip install git+https://github.com/lukauskas/dtwco.git

potentially some others

2. compile and copy the compiled lib out

cd extension-ffi/script/

python build_dp.py

cp -r _ext/ ../../

----------- run -----------

1. change data locations "dir=/path/to/UCR_dataset" in configs/config_*.py and test_barycenter.py

2. change hyper-parameters settings if needed

3. to run sample dtw learning:

python3 main.py config_sample 0 0  # here only config_*, don't put extension .py

or to run barycenter experiment:

python3 test_barycenter.py 0 0 dataset_name(e.g. yoga) 

or run cnn:

python3 main.py config_cnn 0 0

