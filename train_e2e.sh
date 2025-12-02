mlflow server --host 127.0.0.1 --port 8080 &
python train.py --config-name convnet train.epochs=4 data.batch_size=16 # сделайте батч меньше, если видюха не тянет
