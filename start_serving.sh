#!/bin/bash

# install tensorflow repo
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
	curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
# install tensorflow serving tools
sudo apt update
sudo apt install tensorflow-model-server
pip3 install requests
python3 convert_model.py
saved_model_cli show --dir ./openpose/1 --all # checkout output model
saved_model_cli run --dir ./openpose/1 --tag_set serve --signature_def serving_default --input_exp 'input_1=np.random.normal(size=(1,128,128,3))' # test serving
# start serving at background
cp -rv ./openpose /tmp # serving model needs an absolute path
nohup tensorflow_model_server --rest_api_port=8503 --model_name=openpose --model_base_path="/tmp/openpose" >server.log 3>&1 &
