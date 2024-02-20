poetry build

copy .\dist\my_ml_util-0.1.0-py3-none-any.whl .\kaggle_dataset

xcopy .\my_ml_util .\kaggle_dataset\my_ml_util /e /i /h /Y

cd ./kaggle_dataset

kaggle datasets version --dir-mode zip -p ./ -m "API Upload Wheel + src"

cd ..