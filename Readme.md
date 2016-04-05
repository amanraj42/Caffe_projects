# Patch wise classification of regions in a comic image to detect the text portion	

`sh ru_training.sh`
runs training and creates the log

`implement_text.detect.py`
run the trained caffe_model

`hdf52leveldb_siamese.cpp`
used for conversion of data into LevelDB format

`.protoxt`
files containing network definition and training parameters
