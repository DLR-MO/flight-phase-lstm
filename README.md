# Flight Phase Identification from Trajectory with LSTM

Flight phase estimator for trajectory data based on K-means clustering and LSTM.
The following flight phases are identified based on the ICAO ADREP standard.


phase used for classification |ICAO primary phase| ICAO sub-phase |
---|---|---|
taxi | taxi | all
take-off | take-off | take-off run
initial climb | take-off | initial climb
climb | en route | climb to cruise
cruise | en route | cruise & change of cruise level
descent | en route | descent
approach | approach | all
landing | landing | level off-touchdown & landing roll 

Training data inlcludes trajectory (Altitude, Speed, Rate of Climb) and flight phases found with the flight phase finder tool on X-plane simulator data. 
After training only trajectory data is required to estimate flight phase in order to transfer the model to ADS-B data.

## Requirements
The list of requirements can be found in the requirements.txt file.

Main requirements:
- Pytorch
- Numpy
- Matplotlib

## Using pretrained model on custom ADS-B data

### 1) Preprocess ADS-B data

```bash
python src/ADSB_preprocessing.py --folder custom_files_folder
```

If no storage folder is given a new folder is created with the same name as the original folder and '_preprocessed' appended
The results of the preprocessing are the files themselves, the images that compare before and after and the reports on the quality.

In order to obtain reports on the quality of the analysed flights an overview file has to be provided.

For more options and personalisation see
```bash
python src/ADSB_preprocessing.py --help
```

### 2) Run evaluation on preprocessed flights

```bash
python src/evaluation.py --custom_data_path custom_files_folder_preprocessed/csvs
```

This stores the images of the labeled flights together with the CSV files that include the labels in the results folder.

## Training a model

If one wishes to train a new model on their own FDR data a dataset can be created:
- Use the ```find_flight_phases(pandas_df)``` function in src/pof_functions/flight_phase_finder_core.py to label each flight.\
(For X-plane data  ```python src/pof_functions/flight_phase_finder_xplane.py``` takes raw x-Plane txt log files from the data/xplane_raw folder and separates, labels and stores them.)
- Store trajectory data seperate from its labels respectively in data/preprocessed/trajectories_train and data/preprocessed/labels_train
- ```python src/pof_functions/create_dataset.py``` takes the preprocessed files and generates a training and test dataset for training.
  
The src/parameter_girdsearch.py module allows to train different models with different hyperparameters in parallel each in its own terminal. Please consult:

```bash
python src/parameter_gridsearch.py --help
```

for details.


## Contributors
<a href="https://github.com/EmyArts-DLR">Emy Arts</a>

<a href="https://github.com/AlexanderKamtsiuris">Alexander Kamtsiuris</a>
