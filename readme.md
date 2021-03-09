# SmartShop1.0  Developer documentation

## brief introduction
Model training is performed on popular deep learning frameworks (Qualcomm Neural processing SDK supports Caffe, Caffe2, ONNX, and Tensorflow models.) After the training is completed, the trained model is converted into a DLC file, which can be loaded into the Qualcomm Neural processing SDK runtime.
Users can use Qualcomm Neural processing SDK tool to convert the trained model to DLC file, then use one of the Snapdragon accelerated computing cores to use this DLC file to perform the forward inference process.

This project mainly introuduces how to use the Qualcomm Neural processing SDK tool to convert the age-gender-estimation model to DLC to show the ability of model conversion and prepare DLC file for SmartShop2.0 based C865DK

## The main development process:
1. Set up the development environment
2. Show age-gender-estimation
3. Convert to DLC file

Please refer to <Project Dir>/doc/Detailed_introduction_of_model_conversion.docx for detail.

