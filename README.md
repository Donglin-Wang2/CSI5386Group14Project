# More Baseline for VQA
## Data Folder Structure
For consistency, the folders from which you access data should look like this:
```
.
├── README.md
├── data_download.sh
├── dataset_test.ipynb
├── notebook.ipynb
├── preprocessing.ipynb
├── processed_data          # Contains all the processed files from raw_data
│   ├── questions.npy
│   └── train_data_ids.npy
├── raw_data                # Contains the original, unprocessed data
│   ├── train2014
│   ├── train2014.zip
│   ├── v2_OpenEnded_mscoco_train2014_questions.json
│   └── v2_mscoco_train2014_annotations.json
└── vqa.py
```