## Ojbect  
This version is modified for differential privacy on the basis of V2 code.   
Mainly modified file: `CNN-readmission-inference-deployment.py`, `param_process.py` and `CNN-readmission-inference.py`.  


## Project structure:
```
readmission_CNN_V3
├──checkpoints/
├──inputfolder/
│	├── ccs_icd10cm_2017.csv
│	├── DRG.pkl
│	└── mysql_preprocess.py
├──outputfolder/
│	├── meta_data.csv
│	├── readmitted_prob.csv
│	└── true_label_prob_matrix.csv
├──visfolder/
├──CNN-inference.sh
├──CNN-readmission-inference-deployment.py
├──CNN-readmission-inference.py
├──CNN-readmission-trainvalidationtest.py
├──CNN-trainvalidationtest.sh
├──data_loader.py
├──evaluate_metric.sh
├──explain_occlude_area.py
├──explain_occlude_area_format_out.py
├──healthcare_metrics.py
├──insert_into_table.py
├──mapping_dict.pkl
├──model.py
├──mynet.py
├──param_process.py
├──ranked_readmission_feature_mapping_explanation.txt
├──readmission-feature-mapping-explanation.txt
└──readme.txt

```


<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>CNN-readmission-trainvalidationtest.py</td><td>Training phase file</td>
    </tr>
    <tr>
        <td>CNN-readmission-inference.py</td><td>Inference phase file</td>
    </tr>
    <tr>
        <td>CNN-readmission-inference-deployment.py</td><td>Deployment on NUH(insert into database compared with inference phase file)</td>
    </tr>
    <tr>
        <td>data_loader.py</td><td>The code for data loading</td>
    </tr>
    <tr>
        <td>evaluate_metric.sh</td><td>The code for models comparison</td>
    </tr>
    <tr>
        <td>explain_occlude_area.py</td><td>explain occlude area</td>
    </tr>
    <tr>
        <td>explain_occlude_area_format_out.py</td><td>explain occlude area</td>
    </tr>
    <tr>
        <td>healthcare_metrics.py</td><td>The code for metric calculation</td>
    </tr>
    <tr>
        <td>insert_into_table.py</td><td>The code for database insertation(deployment phase)</td>
    </tr>
    <tr>
        <td>mapping_dict.pkl</td><td>feature mapping dictionary when processing the whole dataset</td>
    </tr>
    <tr>
        <td>model.py</td><td>The code to create model</td>
    </tr>
    <tr>
        <td>mynet.py</td><td>The model design and construction code</td>
    </tr>
    <tr>
        <td>param_process.py</td><td>The code for processing the parameters(especially gradient)</td>
    </tr>
    <tr>
        <td>ranked_readmission_feature_mapping_explanation.txt</td><td>feature mapping dictionary when processing the whole dataset</td>
    </tr>
    <tr>
        <td>readmission-feature-mapping-explanation.txt</td><td>feature mapping dictionary when processing the whole dataset</td>
    </tr>
</table>


## Steps

### preprocess data

#### Usage
Go into readmission_CNN_code/inputfolder/ folder, execute command
```bash
python mysql_preprocess.py
```

* Read data from csv files: function `data_saver(data_src, data_des, used_for_test = False)` is to read data from csv files. In our settings, given the data source path and the path of demographic, inpatient and outpatient files, feature, label and patients files for training and test would be generated to data destination.  
* Read data from MySQL database: function `process_new_data(data_src, data_des, mapping_file, days=20, used_for_test=True)` is to read data from MySQL database, preprocess and generate several files to data destination. You can change the "user", "password", "host" and "database" in function `process_new_data`. In out settings, two files generated are "deployment_patients_id.txt" and "deployment_features.txt", which save patient No. and their corresponding features separately for each patient.  


#### Parameters:   
* `data_scr`: list, the name of the data sources (demographic, inpatient, outpatient)  
* `data_des`: list, the name of the data destination. e.g, [feature_file, label_file, patient_file] for `data_saver()`, [feature_file, patient_file] for `process_new_data()`.  
* `mapping_file`: str, the path of feature mapping file.  
* `days`: int, the number of days of data we read from database.  
* `used_for_test`: boolean, whether the results for test usage.  

#### intermediate results
When reading data from csv files, we will get several intermediate results in `inputfolder/`.    
* dataset.pkl: 3-tuple in pkl file saving demographic, inpatient and outpatient DataFrames.  
* joinedTable_df.pkl: One DataFrame in pkl file after joining demograhic, inpatient and outpatient data.  
* df.pkl: Final DataFrame in pkl file after calculating other features(e.g., #Invisits) from joinedTable.   
* DRG_dict.pkl: pkl file saving DRG codes and their corresponding descriptions, generated from file ccs_icd10cm_2017.csv. e.g., {T364X3D {'DESCRIPTION': 'Poisoning by tetracyclines, assault, subsequent encounter'}}  
* patient_DRG_info.pkl: dict in pkl file saving patients' IDs and their corresponding DRG. e.g., {KPDUMNXHGPx345: {'DRG': 'L271|M4782|M542|R208|R42|Y451|Y9222'}}

#### final results
When reading data from csv files, features, labels and patients' ID would be saved into separate files.  
When reading data from MySQL database, only features and patients' ID would be saved.



### training phase
Under `readmission_CNN_code/` folder, execute command:
```bash
python CNN-readmission-trainvalidationtest.py -inputfolder 'inputfolder' -outputfolder 'outputfolder' -visfolder 'visfolder'
```
This would train the model designed in `mynet.py`, and save checkpoint into `checkpoints/` folder.


### testing phase
Under `readmission_CNN_code/` folder, execute command:
```bash
python CNN-readmission-inference.py -inputfolder 'inputfolder' -outputfolder 'outputfolder' -visfolder 'visfolder'
```
This would test the model saved `checkpoints/` folder, and generate related files in `outputfolder/` folder.


### deployment phase
Under `readmission_CNN_code/` folder, execute command:
```bash
python CNN-readmission-inference-deployment.py -inputfolder 'inputfolder' -outputfolder 'outputfolder' -visfolder 'visfolder'
```
This command would read the two files generated from preprocessing step, and insert predicted records into table under specific database. You can change the "user", "password", "host", "database" and "tablename" at the end of function 'train'.


### Evaluation models
The files generated during testing phase under `outputfolder/`, you can command under this folder:
```bash
cp ../evaluate_metric.py .
python evaluate_metric.py
```
Then a file `model_comparison.json` would be generated to show the `AUC`, `Accuracy`, `Sensitivity` and `Specificity` of the model within the same folder.



## Our results 
clip_gaussian_checkpoints/ and clip_gaussian_outputfolder/ are invalid, cause the threshold setting is not reasonable.  
gaussian_variance_checkpoints/ and gaussian_variance_outputfolder/ save model checkpoints and readmitted_probability when test model metric respectively, the models are trained with gaussian noises. Check readme.txt under folder.    
checkpoints/ and outputfolder/ save model and readmitted_probability without adding gaussian noises.   
