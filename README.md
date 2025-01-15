# learnearthPRJ
application of ia tools on an earthquake dataset following a problematic. 

the code is structured as following:

inside the parent container, you can find all the .py files to make the workflow.
- cleaning.py allows to isolate the columns from the original database.csv
- treating_data.py is a script to create a new variable (population_impacted) by using overpass API
- clustering.py is the application of algorithm to separate the data in order to classify the different earthquackes. 
- classify.py is the execution of the supervised classification training using the clustering output. 

Then all the results, graphics or databases are scattered among the different folders:

datasets is a folder containing all the data used before and after clusterisation. including the original database, the cleaned version, shorter version with only the first 1000 data. and the clustering database output with the new categorisations. 

clustering_results contains the visualisations of the clustering, boxplots per cluster and PCA in 2 dimensions. Also the txt files for the clustering output to search for incoherences. 

Finally the classification results are in 2 different folders. 
The first one is 1000values, results of the training on the shorter version using 1000 data. 
    - confusion matrixes
    - AUCs
    - hyperparameters impact 
for each of the trained models.

Same for certains of the models that have been trained on all the data (23000) in the folder Allvalues.