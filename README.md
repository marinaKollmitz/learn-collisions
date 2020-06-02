# Predicting Obstacle Footprints from 2D Occupancy Maps by Learning from Physical Interactions

Sensors are expensive, so service robots in human-centered environments have to learn to cope with incomplete sensor information. In this repository you find our code and dataset for teaching mobile robots to predict where collisions may occur in 2D occupancy maps from planar laser rangefinders. 

If you are using our code or dataset in your research, please consider citing our paper:

```
@INPROCEEDINGS{kollmitz20icra,
  author = {Marina Kollmitz and Daniel B\"uscher and Wolfram Burgard},
  title = {Predicting Obstacle Footprints from 2D Occupancy Maps by Learning from Physical Interactions},
  booktitle = {Proc.~of the IEEE Int.~Conf.~on Robotics \& Automation (ICRA)},
  year = {2020},
  url = {http://ais.informatik.uni-freiburg.de/publications/papers/kollmitz20icra.pdf}
}
```

## Get the code and dataset

1. Clone repository, which includes the code and dataset. ```$LEARN_COLLISIONS_HOME``` is a placeholder for the directory in which you want to clone the repository:
```
cd $LEARN_COLLISIONS_HOME
git clone https://github.com/marinaKollmitz/learn-collisions/
```
2. Unpack the dataset:
```
cd $LEARN_COLLISIONS_HOME/datasets
tar -xzf SceneNetCollision.tar.gz
```
3. (optional) We suggest to create a virtual python environment:
```
cd $LEARN_COLLISIONS_HOME
virtualenv -p python3 ./venv
source venv/bin/activate
```
4. Install requirements

  4.1. Install pytorch. Follow the installation instructions from https://pytorch.org/ to install pytorch for your system and architecture. The standard will be:
```
pip install torch torchvision
```
  4.2. Install other requirements with pip:
```
pip install PyYAML scipy>=1.2.0 opencv-python matplotlib sklearn
```

## Segment Occupancy Maps
Use the ```segment_map.py``` script to segment an occupancy map with a trained model. Usage:
```
cd $LEARN_COLLISIONS_HOME/scripts
python segment_map.py map_yaml network_dir
```
Example:
```
python segment_map.py ../datasets/SceneNetCollision/occupancy_maps/1Bedroom_SceneFiles_occ.yaml ../networks/dilated_fcn_lenet-60/
```
You should see a heatmap plot of the segmented occupancy map. Use the ```--save``` option to save the map image to the model directory.
