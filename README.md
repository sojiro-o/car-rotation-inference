# 車の回転角予測

---

### 使い方

```
pip install -r requirements.txt
cd src
python predict.py <image_path>
```

予測結果は __予測ラベルのリスト(確率が高い順)__ と __それぞれの確率のリスト__ を返す

---

### test
datasetは以下を想定

```
dataset_dir
	├── 0
	│   └──*.jpg
	├── 1
	├── 2
	...
	└── 7
```

config.yamlの`dataset_path`を変えて`python test.py`  
accrancyを返す

docker build -t car-rotation -f docker/Dockerfile .
docker run -it --gpus all --name car-rotation car-rotation