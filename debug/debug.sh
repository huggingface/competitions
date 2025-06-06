# change this to your HF token if you accessing private repos
# make sure to pass when running docker HF_TOKEN=mytoken 
# export SAFE_DATASET_REPO=dsf-sandbox/video-challenge-debug
# # change this to your model that you want to test
# export MODEL_REPO=safe-challenge/safe-example-submission
export DATASET_PATH=/tmp/data
export MODEL_PATH=/tmp/model
export HF_HUB_ENABLE_HF_TRANSFER=1

# set -ex

cd /app/debug

echo "downloading dataset"
python download_dataset.py

echo "downloading model"
python download_model.py

echo "installing requirments"
python conda run -p $CONDA_ENV_MODEL pip install -r $MODEL_PATH/requirements.txt


echo -e "disabling network"
cp /app/sandbox $MODEL_PATH/sandbox
chmod 755 $MODEL_PATH/sandbox
chown $UID:$(id -g) $MODEL_PATH/sandbox

cd $MODEL_PATH 
echo "running script.py"
cd $MODEL_PATH
sandbox conda run -p $CONDA_ENV_MODEL python script.py

echo "output file"
cat "submission.csv"


cd /app/debug
echo "evaluating"
python eval.py


