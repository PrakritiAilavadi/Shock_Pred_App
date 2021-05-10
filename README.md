# Shock_Pred_App
SAFE ICU Shock Prediction Flask App for health practitioner's usability 

Shock Prediction App ShoQPred is Deployable Human Centric Software Suite for hemodynamic shock prediction which leverages AI and ML methods on physiological vitals time-series data to predict hemodynamic shock prediction upto 3 to 12 hours before the event.


## Requirements
1. Python 3.6+ / Django
2. Tsfresh
3. Sklearn (extended requirements in requirements.txt)
4. Flask
5. Keras, Tensorflow
6. Docker for Docker deployment


## Summary of main_app.py 

It is the main python flask file that drives the application, links to the HTML templates and contains the pickle dumped models in the Subdirectory folder.

<br> Functions included in this file are:

1. <b> allowed_file </b>: To check if the uploaded file has the allowed extensions.
2. <b> transform </b>: To transform text file contents in the required form
3. <b> index </b>: To render the homepage template
4. <b> start </b>: To go to the start modelling page on clicking the start modelling button on the homepage. It fetches the user selected disease to predict (AHE/LOS/Shock) and the model which would be used to predict (RF/LSTM/DenseNets) and correspondingly renders the template for the same.
5. <b> authors </b>: Renders the static authors template to display the details of authors and co-authors of the application.
6. <b> showPreprocessing </b>: display the pre-processed features if already loaded
7. <b> showFeatures </b>: display features if already loaded
8. <b> showClassification </b>: display prediction result if already loaded
9. <b> upload_densenet_file </b>: for uploading single file for DenseNet based prediction. Fetches the prediction variables selected by user and sends the uploaded file to <b>startDensenets.html template</b> for displaying on the main screen.
10. <b> uploaded_file </b>: retrieve files from the directory
11. <b> uploaded_files_with_age_gender </b>: fetch the user uploaded patient files with age and gender variables included. Select specific columns for every disease. Concatenate al patient files and send them to <b>startModelling.html template</b> for displaying on the screen.
12. <b> upload_folder </b>: for carrying out the same process as point 11 just without age and gender. Followed by rendering the output concatenated files to <b>startModelling.html template</b>
13. <b> calculate_features </b>: extract ts-fresh features from the uploaded file. Impute the features. Add the age column in the features if age-gender files are selected by user. Then pass the features to display it on the next page with is <b>features.html template </b>
14. <b> pre_process </b>: pre-process the extracted boruta/ts-fresh features by a. mean centering for shock, b. normalizing for AHE and LOS. Finally do snake representation* and pass it to <b> process_snake.html tenplate </b>
15. <b> prediction </b>: The following steps are followed for prediction: <br>
<t> 1. If the model chosen is densenet: Load the processed features and statistics of the trained set (mean and std) from the Subdirectory. Normalize the processed features of the uploaded user dataset. Next load the densenet prediction pre-trained model from the subdirectory. Use it to make prediction on the processed and normalized features. Finally put those prediction in a new clean dataframe format and pass it to <b> densenet_prediction.html template </b> to display it on the final prediction page  </t>
<br><t> 2. If the model chosen is random forest and to-predict disease is shock: </t>

#### *The Snakemake workflow management system is a tool to create reproducible and scalable data analyses.
