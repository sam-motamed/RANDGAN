# RANDGAN for semi supervised detection of COVID-19 in Chest X-rays

RANDGAN is a generative adversarial model implemented in Python for classification of COVID-19 Positive and COVID-19 Negative Chest X-rays.

If you use our model or the segmented COVIDx dataset,
Please cite our paper  https://arxiv.org/abs/2010.06418

* To generate the segmented COVIDx dataset, please refer to 
	https://github.com/IMICSLab/Covidx-IMICS-Lung-Segmentation
```
* Your directory structure should be as follows:
.
├── RANDGAN_model.py
├── main.py
├── result     #generated images at each iteration are saved here
├── weight
├── modified   #direcvtory where anomaly scores are saved in
├── data                    
│   ├── COVID_train.npy              
│   ├── Normal_train.npy              
│   ├── Pneumonia_train.npy             
│   └── ...                #train and etst numpy arrays
└── ...
```

# Train the model

* By running main.py (setting line 23 to train), model starts training

* make sure you change the file / directory paths to your local environment 
# Test the model

* By setting main.py to test mode (line 23), you can load in test images and calculate anonmaly scores for each image (saved to modified folder as CSV file.

## Requirements

```bash
The main requirements are listed below:

Tested with Keras 2.3.1
Python 3.6
OpenCV 3.4.2
scikit-image 0.16.2
Numpy
Scikit-Learn
Matplotlib
```
## RANDGAN Contributors
* iMICS Lab, University of Toronto, Canada https://imics.ca/
	* Saman Motamed
	* Farzad Khalvati
	* Ernest Khashayar Namdar
	* Patrik Rogalla
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
