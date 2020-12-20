# openpose-serving-tf2.0

This project implement the serving code for [project](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)

## prerequisite tools

install prerequisite tools with the following cmds

```bash
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
	curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt update
sudo apt install python3-opencv tensorflow-model-server
pip3 install numpy tf-nightly-2.0-preview requests
```

## download trained model

The trained mode is adopted from michalfaber's project mentioned above. It can be downloaded [here](https://pan.baidu.com/s/1p7jsvN83yZIXCv5egRWFjw), passcode is *88t6*.

## convert the model from hdf5 to saved model

With hdf5 model placed in the current directory, convert it with the following command.

```python
python3 convert_model.py
```

Then you can find openpose directory presenting at current directory.

## start serving

With hdf5 model placed in the current directory, start serving the model with following command

```python
bash start_serving.sh
```

## predict with model server

After successfully executed start_serving.sh, you can try to do pose estimation through the model server with the following command.

```python
python3 Predictor.py
```

The script will estimate the pose of humans in the test image. you can try other images by changing image path at line 263 of the script.

