# LSTM-flight-phase-estimator

Flight phase estimator for trajectory data based on K-means clustering and LSTM.

Project under supervision of Hendrik Meyer.

With the help of the flight phase finder of Alexander Kamtsiuris, this project aims at identifying flight phases from trajectory data. 
Training data inlcludes trajectory (Altitude, Speed, Rate of Climb) and flight phases found with the flight phase finder tool on X-plane simulator data. 
After training only trajectory data is required to estimate flight phase in order to transfer the model to ADS-B data.

## Requirements
The list of requirements can be found in the requirements.txt file.

Main requirements:
- Pytorch
- Numpy
- Matplotlib

## Training a model

The src/parameter_girdsearch.py module allows to train different models with different hyperparameters in parallel each in its own terminal. Please consult:

```bash
python src/parameter_gridsearch.py --help
```

for details.

## Using pretrained model on custom ADS-B data

1) Preprocess ADS-B data:

```bash
python src/ADSB_preprocessing.py --folder custom_files_folder --overview_file flight_log_file.csv
```

If no storage folder is given a new folder is created with the same name as the original folder and '_preprocessed' appended
The results of the preprocessing are the files themselves, the images that compare before and after and the reports on the quality.

2) Run evaluation on preprocessed flights

```bash
python src/evaluation.py --custom_data_path custom_files_folder_preprocessed/csvs
```

This stores the images of the labeled flights together with the CSV files that include the labels in the results folder.
