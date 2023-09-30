## Learning Cross-Modality Interaction for Robust DepthPerception of Autonomous Driving
</br>
This is the Pytorch implementation of our work on depth completion.

## Environment
1. Python 3.6
2. PyTorch 1.2.0
3. CUDA 10.0
4. Ubuntu 16.04
5. Opencv-python
6. pip install pointlib/.

## Dataset
[KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
Prepare the dataset according to the datalists (*.txt in datasets)
```
datasets
|----kitti 
    |----depth_selection 
        |----val_selection_cropped
            |----...
        |----test_depth_completion_anonymous   
            |----...     
    |----rgb     
        |----2011_09_26
        |----...  
    |----train  
        |----2011_09_26_drive_0001_sync
        |----...   
    |----val   
        |----2011_09_26_drive_0002_sync
        |----...
```
[CARLA](https://github.com/carla-simulator/carla)

## Training
python train_prediction.py

## Visualization
python visualize.py

## Contact
oak-chen@mail.nwpu.edu.cn
