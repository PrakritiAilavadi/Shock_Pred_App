from flask import Flask, request, render_template, url_for, redirect, send_file, make_response
from werkzeug.utils import secure_filename
from flask import send_from_directory, current_app
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
import io
import pandas as pd
import numpy as np
import glob
import csv
import unittest
import keras
from keras import backend as k_backend
import tensorflow
import pickle
import joblib
import os
import tsfresh
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features
from sklearn.model_selection import train_test_split
import sys
import SubDirectory.script
import SubDirectory.densenet_predictions
from wtforms.validators import InputRequired, Length

# 1 is abnormal
# 0 is normal

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/prakritiailavadi/PycharmProjects/test1/example.db'
# db = SQLAlchemy(app)
#
#
# class CsvUpload(db.Model):
#     __tablename__ = "csv_uploads2"
#     id = db.Column(db.String(300), primary_key=True)
#     label = db.Column(db.String(300))
#     GENDER = db.Column(db.String(300))
#     age = db.Column(db.Integer)
#
#
# def __init__(self, id, label, GENDER, age):
#    self.id = id
#    self.label = label
#    self.GENDER = GENDER
#    self.age = age


UPLOAD_FOLDER = '/Users/prakritiailavadi/PycharmProjects/test1/folder'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FLASK_DEBUG = 1


# flag is empty for folder upload and is equal to file_upload for single file upload
flag = ""
pre_var = ""
fileFeatures = ""
result = ""
to_predict = ""
model = ""
dn_filename = ""
age_column = []
gender_column = []


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


# homepage
@app.route("/")
def index(user=None):
    # var = pickle.load(open('ahe_rf_age_gender.sav', 'rb'))
    return render_template("homepage.html", user=user)


# open start modelling main page
@app.route("/start", methods=['GET', 'POST'])
def start():
    # shock/ahe
    # SubDirectory.script.printfunction("Hello world")
    if request.method == 'POST':
        global to_predict
        global model
        to_predict = request.form['to_predict']
        model = request.form['model']

    # if model is of dense nets
    if model == 'dn':
        # if file already uploaded then show
        if flag == "file_upload" or flag == "folder_upload":
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], var))
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            return render_template("startDensenets.html", data_frame=df.to_html())
        return render_template("startDensenets.html")

    # if model is of random forest
    if model == 'rf':
        # if file already uploaded then show
        if flag == "file_upload" or flag == "folder_upload":
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], var))
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            return render_template("startModelling.html", data_frame=df.to_html())
        return render_template("startModelling.html")


# authors page
@app.route("/authors")
def authors():
    return render_template("authors.html")


# show processed features if there
@app.route("/showPreProcessing", methods=['GET', 'POST'])
def showPreProcessing():
    if model == 'dn':
        if pre_var == "Processed_Dataset.npy":
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], pre_var))
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            return render_template("process_snake.html", data_frame=df.to_html())
        return render_template("process_snake.html")
    if pre_var == "Processed_Dataset.csv" and model == "rf":
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], pre_var))
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        return render_template("pre_processing.html", data_frame=df.to_html())
    return render_template("pre_processing.html")


# show features if there
@app.route("/showFeatures", methods=['GET', 'POST'])
def showFeatures():
    if fileFeatures == "features.csv":
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        return render_template("features.html", features=df.to_html())
    return render_template("features.html")


# show predicted results if there
@app.route("/showClassification", methods=['GET', 'POST'])
def showClassification():
    if model == 'dn':
        if result == "prediction_result.csv":
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            return render_template("densenet_prediction.html", result=df.to_html())
        return render_template("densenet_prediction.html")
    if model == "rf":
        if result == "prediction_result.csv":
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            return render_template("classification.html", result=df.to_html())
        return render_template("classification.html")

## for uploading single file for densenet
@app.route('/upload_densenet_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        file = request.files['file']
        filename = secure_filename(file.filename)

        hr = request.form['hr_options_file']
        resp = request.form['resp_options_file']
        spo2 = request.form['spo2_options_file']
        bp = request.form['bp_options_file']
        bp_dias = request.form['bp_dias_options_file']

        data2 = df.copy()

        hr_column = data2.iloc[:, int(hr) - 1]
        resp_column = data2.iloc[:, int(resp) - 1]
        spo2_column = data2.iloc[:, int(spo2) - 1]
        final_abp_sys_column = data2.iloc[:, int(bp) - 1]
        final_abp_dias_column = data2.iloc[:, int(bp_dias) - 1]

        list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column, final_abp_dias_column]
        data3 = pd.DataFrame(data=list_of_columns).T
        data3.insert(0, 'ID', filename)

        normal_file_data_frame = pd.DataFrame(data3)
        # renaming columns
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[1]: "X.HR."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[2]: "X.RESP."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[3]: "X.SpO2."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[4]: "final_abp_sys"},
                                      inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[5]: "final_abp_dias"},
                                      inplace=True)

        global dn_filename
        dn_filename = filename
        global var
        var = filename
        normal_file_data_frame.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global flag
        flag = "file_upload"
        return render_template("startDensenets.html", data_frame=normal_file_data_frame.to_html())


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# for uploading files with gender & age
@app.route('/upload_files_with_age_gender', methods=['POST'])
def upload_files_with_age_gender():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        list_of_normal_files = uploaded_files
        number_of_files = len(list_of_normal_files)
        print(number_of_files)
        normal_file_list = []
        filenames_array = []
        hr = request.form['hr_options_file']
        resp = request.form['resp_options_file']
        spo2 = request.form['spo2_options_file']
        bp = request.form['bp_options_file']
        bp_dias = request.form['bp_dias_options_file']

        age_gender_file = request.files['file_age_gender']
        filename_age_gender = secure_filename(age_gender_file.filename)
        patient_id = request.form['id_options_file']
        gender = request.form['gender_options_file']
        age = request.form['age_options_file']



        j = 0
        global flag
        flag = "folder_upload"
        for f in list_of_normal_files:
            k = f.filename
            data1 = pd.read_csv(f)
            data2 = data1.copy()
            hr_column = data2.iloc[:, int(hr)-1]
            resp_column = data2.iloc[:, int(resp)-1]
            spo2_column = data2.iloc[:, int(spo2)-1]
            final_abp_sys_column = data2.iloc[:, int(bp)-1]
            final_abp_dias_column = data2.iloc[:, int(bp_dias) - 1]
            # colname = data2.columns[int(bp)-1]

            # Creating a list of the series':
            if to_predict == 'shock':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column, final_abp_dias_column]
            elif to_predict == 'ahe':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column]
            elif to_predict == 'los':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column]

            data3 = pd.DataFrame(data=list_of_columns).T

            # Adding another column for the patient ID:
            data3.insert(0, 'ID', k)
            # data3.insert(4, 'BP', final_abp_sys_column)
            normal_file_list.append(data3)

            # add filenames to a list say filenames_array
            filenames_array.append(k)
            j = j + 1

        # getting list of filenames of the files uploaded and removing extensions
        filenames_without_extension = [os.path.splitext(x)[0] for x in filenames_array]
        print(filenames_without_extension)

        global age_column
        global gender_column

        # reading the file input by user for age and gender details
        age_gender_df = pd.read_csv(age_gender_file)
        # extracting specific filenames from the age_gender_df
        age_gender_df_copy = age_gender_df[age_gender_df.iloc[:, int(patient_id) - 1].isin(filenames_without_extension)]
        patient_id_column = age_gender_df_copy.iloc[:, int(patient_id) - 1]
        # getting gender column and replacing gender by 0 for male and 1 for female
        gender_column = age_gender_df_copy.iloc[:, int(gender) - 1].tolist()
        gender_column = [0 if x == 'M' else 1 for x in gender_column]
        # getting age column and replacing age > 90 by 90
        age_column = age_gender_df_copy.iloc[:, int(age) - 1].tolist()
        age_column = np.array(age_column)
        age_column[age_column > 90] = 90

        print(patient_id_column)
        print(age_column)
        print(gender_column)

        normal_file_data_frame = pd.concat(normal_file_list, ignore_index=True)
        normal_file_data_frame = pd.DataFrame(normal_file_data_frame)

        # renaming columns
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[1]: "X.HR."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[2]: "X.RESP."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[3]: "X.SpO2."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[4]: "final_abp_sys"},
                                      inplace=True)
        if to_predict == 'shock':
            normal_file_data_frame.rename(columns={normal_file_data_frame.columns[5]: "final_abp_dias"},
                                          inplace=True)

        global var
        filename = "concatenated_dataset.csv"
        var = filename
        normal_file_data_frame.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template("startModelling.html", data_frame=normal_file_data_frame.to_html())


# for uploading files without gender & age
@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        list_of_normal_files = uploaded_files
        normal_file_list = []
        hr = request.form['hr_options']
        resp = request.form['resp_options']
        spo2 = request.form['spo2_options']
        bp = request.form['bp_options']
        bp_dias = request.form['bp_dias_options']

        j = 0
        global flag
        flag = "folder_upload"
        for f in list_of_normal_files:
            k = f.filename
            data1 = pd.read_csv(f)
            data2 = data1.copy()

            hr_column = data2.iloc[:, int(hr)-1]
            resp_column = data2.iloc[:, int(resp)-1]
            spo2_column = data2.iloc[:, int(spo2)-1]
            final_abp_sys_column = data2.iloc[:, int(bp)-1]
            final_abp_dias_column = data2.iloc[:, int(bp_dias) - 1]
            # colname = data2.columns[int(bp)-1]

            # Creating a list of the series':
            if to_predict == 'shock':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column, final_abp_dias_column]
            elif to_predict == 'ahe':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column]
            elif to_predict == 'los':
                list_of_columns = [hr_column, resp_column, spo2_column, final_abp_sys_column]

            data3 = pd.DataFrame(data=list_of_columns).T

            # Adding another column for the patient ID:
            data3.insert(0, 'ID', k)
            # data3.insert(4, 'BP', final_abp_sys_column)
            normal_file_list.append(data3)
            j = j + 1

        normal_file_data_frame = pd.concat(normal_file_list, ignore_index=True)
        normal_file_data_frame = pd.DataFrame(normal_file_data_frame)

        # renaming columns
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[1]: "X.HR."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[2]: "X.RESP."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[3]: "X.SpO2."}, inplace=True)
        normal_file_data_frame.rename(columns={normal_file_data_frame.columns[4]: "final_abp_sys"},
                                      inplace=True)
        if to_predict == 'shock':
            normal_file_data_frame.rename(columns={normal_file_data_frame.columns[5]: "final_abp_dias"},
                                          inplace=True)

        global var
        filename = "concatenated_dataset.csv"
        var = filename
        normal_file_data_frame.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template("startModelling.html", data_frame=normal_file_data_frame.to_html())

# calculate tsfresh features from the concatenated dataset
@app.route("/calculate_features")
def calculate_features():
    # if flag == "file_upload":
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], var))
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    features = extract_features(df,  column_id='ID')         ## dataset needs a column id to be there
    features = impute(features)
    # add age column if age_gender selected by user and hence if length of age > 0
    if len(age_column) > 0:
        features["age"] = age_column
    global fileFeatures
    fileFeatures = "features.csv"
    features.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
    return render_template("features.html", features=features.to_html())


# process the extracted features
@app.route("/pre_process")
def pre_process():
    if model == 'rf':
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

        df.columns = df.columns.str.replace("(", ".")
        df.columns = df.columns.str.replace(")", ".")
        df.columns = df.columns.str.replace(", ", "..")
        df.columns = df.columns.str.replace('"', ".")

        # For substracting the mean from each column:
        if to_predict == 'shock':
            if len(age_column) > 0:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary_shock_age.csv"))
            else:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary.csv"))
            for i in np.arange(len(mean_of_all_features.columns)):
                df[mean_of_all_features.columns[i]] = np.subtract(df[mean_of_all_features.columns[i]],
                                                                  mean_of_all_features[mean_of_all_features.columns[i]])
        elif to_predict == 'ahe':
            if len(age_column) > 0:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary_ahe_age.csv"))
                sd_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_sd_summary_ahe_age.csv"))
            else:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary_ahe.csv"))
                sd_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_sd_summary_ahe.csv"))
            for i in np.arange(len(mean_of_all_features.columns)):
                df[mean_of_all_features.columns[i]] = np.subtract(df[mean_of_all_features.columns[i]],
                                                                  mean_of_all_features[mean_of_all_features.columns[i]])
                df[sd_of_all_features.columns[i]] = np.divide(df[sd_of_all_features.columns[i]], sd_of_all_features[sd_of_all_features.columns[i]])
        elif to_predict == 'los':
            if len(age_column) > 0:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary_los_age.csv"))
                sd_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_sd_summary_los_age.csv"))
            else:
                mean_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_summary_los.csv"))
                sd_of_all_features = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_sd_summary_los.csv"))

            for i in np.arange(len(mean_of_all_features.columns)):
                df[mean_of_all_features.columns[i]] = np.subtract(df[mean_of_all_features.columns[i]],
                                                                  mean_of_all_features[mean_of_all_features.columns[i]])
                df[sd_of_all_features.columns[i]] = np.divide(df[sd_of_all_features.columns[i]],
                                                              sd_of_all_features[sd_of_all_features.columns[i]])

        if len(gender_column) > 0:
            df["GENDER"] = gender_column

        global pre_var
        pre_var = "Processed_Dataset.csv"
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], pre_var))
        return render_template("pre_processing.html", data_frame=df.to_html())
    if model == 'dn':
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], var))
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]

        # do snake representation via snake
        array_snake = SubDirectory.script.snake(df)

        pre_var = "Processed_Dataset.npy"
        np.save(os.path.join(app.config['UPLOAD_FOLDER'], pre_var), array_snake)
        return render_template("process_snake.html", array_snake=array_snake)


# prediction done on selected features
@app.route("/prediction")
def prediction():
    if model == 'dn':
        array1 = np.load(os.path.join(app.config['UPLOAD_FOLDER'], pre_var))
        # print(array1)
        x = np.pad(array1, ((0, 0), (7, 7), (7, 7), (0, 0)), mode='constant')
        img_height = 30
        img_width = 30
        mean_calc1 = np.load('mean_shock_image_mimic.npy')
        std_calc1 = np.load('std_shock_image_mimic.npy')

        x -= mean_calc1
        # Apply featurewise_std_normalization to test-data with statistics from train data
        x /= (std_calc1 + k_backend.epsilon())

        predicted_output = SubDirectory.densenet_predictions.predict(x)
        filename = dn_filename
        # new_data = pd.DataFrame(columns=['Patient ID', 'Predicted Label'])
        # new_data["Patient ID"] = filename
        # new_data["Predicted Label"] = predicted_output
        new_data = pd.DataFrame({"Patient ID ": [filename],
                                 " Predicted Label": [predicted_output]})
        print(new_data)
        # new_data = new_data.append({filename: predicted_output})
        global result
        result = "prediction_result.csv"
        new_data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
        return render_template("densenet_prediction.html", result=new_data.to_html())

    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], pre_var))
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    data1 = df.copy()

    if to_predict == "shock" and model == "rf":
        # import and get necessary columns from boruta_train.csv
        original_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_train_summ_shock.csv"))
        if len(age_column) > 0:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'label'])].columns
        else:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'label', 'GENDER', 'age'])].columns
        # selected features dataframe
        data1_selected = data1[list(set(data1.columns) & set(boruta_feature_list))]
        data1_selected_ordered = data1_selected[boruta_feature_list]
        # model1 = pickle.load(open('randforest_model.sav', 'rb'))
        if len(gender_column) > 0:
            model1 = pickle.load(open('shock_rf_age_gender.sav', 'rb'))
            predicted_output = model1.predict_proba(data1_selected_ordered)
            predicted = (predicted_output[:, 1] >= 0.61).astype('int')
        else:
            model1 = pickle.load(open('shock_rf_new.sav', 'rb'))
            predicted_output = model1.predict_proba(data1_selected_ordered)
            predicted = (predicted_output[:, 1] >= 0.61).astype('int')
        # add the label column
        uploaded_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
        new_data = pd.DataFrame(columns=['Patient ID', 'Predicted Label', 'Probability'])
        new_data["Patient ID"] = uploaded_data.iloc[:, 0].T
        new_data["Predicted Label"] = np.asarray(predicted)
        new_data["Probability"] = np.asarray(predicted_output[:, 1])
        result = "prediction_result.csv"
        new_data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
        return render_template("classification.html", result=new_data.to_html())
    elif to_predict == "ahe" and model == "rf":
        original_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_train_summ_ahe.csv"))
        if len(age_column) > 0:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'label'])].columns
        else:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'label', 'GENDER', 'age'])].columns
        # selected features dataframe
        print(len(list(set(data1.columns) & set(boruta_feature_list))))
        data1_selected = data1[list(set(data1.columns) & set(boruta_feature_list))]
        data1_selected = data1_selected.loc[:, ~data1_selected.columns.duplicated()]   # removing duplicated columns
        data1_selected_ordered = data1_selected[boruta_feature_list]
        if len(gender_column) > 0:
            model2 = pickle.load(open('ahe_rf_age_gender.sav', 'rb'))
            predicted_output = model2.predict_proba(data1_selected_ordered)
            predicted = (predicted_output[:, 1] >= 0.59).astype('int')
            print(predicted)
        else:
            model2 = pickle.load(open('ahe_rf_21.2.1.sav', 'rb'))
            predicted_output = model2.predict_proba(data1_selected_ordered)
            predicted = (predicted_output[:, 1] >= 0.59).astype('int')
            print(predicted)
        # add the label column
        uploaded_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
        new_data = pd.DataFrame(columns=['Patient ID', 'Predicted Label', 'Probability'])
        new_data["Patient ID"] = uploaded_data.iloc[:, 0].T
        new_data["Predicted Label"] = np.asarray(predicted)
        new_data["Probability"] = np.asarray(predicted_output[:, 1])
        result = "prediction_result.csv"
        new_data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
        return render_template("classification.html", result=new_data.to_html())
    elif to_predict == "los" and model == "rf":
        original_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "boruta_train_summ_los.csv"))
        if len(age_column) > 0:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'LOS_disc'])].columns
        else:
            boruta_feature_list = original_df[original_df.columns.difference(['id', 'LOS_disc', 'GENDER', 'age'])].columns

        # selected features dataframe
        print(len(list(set(data1.columns) & set(boruta_feature_list))))
        data1_selected = data1[list(set(data1.columns) & set(boruta_feature_list))]
        data1_selected = data1_selected.loc[:, ~data1_selected.columns.duplicated()]  # removing duplicated columns
        data1_selected_ordered = data1_selected[boruta_feature_list]
        if len(gender_column) > 0:
            model3 = pickle.load(open('los_rf_age_gender.sav', 'rb'))
            predicted_output = model3.predict_proba(data1_selected_ordered)
            print(predicted_output)
            predicted = (predicted_output[:, 1] >= 0.3376).astype('int')
            print(predicted)
        else:
            model3 = pickle.load(open('los_rf_new.sav', 'rb'))
            predicted_output = model3.predict_proba(data1_selected_ordered)
            print(predicted_output)
            predicted = (predicted_output[:, 1] >= 0.3264).astype('int')
            print(predicted)
        # add the label column
        uploaded_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], fileFeatures))
        new_data = pd.DataFrame(columns=['Patient ID', 'Predicted Label', 'Probability'])
        new_data["Patient ID"] = uploaded_data.iloc[:, 0].T
        new_data["Predicted Label"] = np.asarray(predicted)
        new_data["Probability"] = np.asarray(predicted_output[:, 1])
        result = "prediction_result.csv"
        new_data.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], result))
        return render_template("classification.html", result=new_data.to_html())


@app.route("/download")
def download():
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), var, as_attachment=True)


@app.route("/download_preprocessed_data")
def download_preprocessed_data():
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), pre_var, as_attachment=True)


@app.route("/download_features")
def download_features():
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), fileFeatures, as_attachment=True)


@app.route("/download_result")
def download_result():
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), result, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
