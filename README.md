# Caffe-pp

本项目是让caffe支持OpenCL，可以方便的在Intel,AMD 显卡上训练。
目前LeNet用的所有层均有OpenCL支持，没有OpenCL支持的层，依然可以训练，
系统会自动使用CPU的实现，但是CUDA和OpenCL不能同时使用。
有对本项目感兴趣的朋友，加QQ一起讨论开发caffe.