# DentalAnomalyDetector

To run on cluster, issue the following command
```bash
qsub -cwd f1.job
```
The output should contain F1 scores along with ROC/PR curve plots for each anomaly. 

main.py runs the training routine
model.py specifies the training routine
datagen.py generates the datasets from the image and excel data provided by Dr. Howe on Argon
