# Classification
Data Classification is a supervised method of machine learning that involves a two step process. 
- ***1)*** The learning step, where a model is created from a training set of data. 
- ***2)*** The classification step, which the model created in step-one is used to predict class labels from a new set of data. 

Although there are many methods for classification, this repository will compare two methods, *Decision Tree Induction* , and *Naїve Bayes*.


#### *The Dataset*
The dataset used for this analysis will be “Student.Portuguese”, (link needed for dataset description). 


#### *Training and Testing Data Process*
In order to complete the two step classification process, the data will need to be split into two independent groups, training and testing data. Initially the data is randomly split to contain a 70/30 ratio for training data and testing data respectively. However, this would mean that the model is trained using only 70 of the data. Any crossover then the model has the potential to be overfitted to the data. To increase the sample data and improve the model training, the multiple *k-fold cross-validation* method was used. More specifically, k-fold was setup to create 10 folds, and then repeat that process 10 times to create 100 different sampling sets of training data. This repeated k-folds are then averaged, this helps ensure a much more reliable performance indication of the generated model [Rafaeilzadeh, Tang and Liu, (2009)][8]. In addition, the data selected for training is stratified to minimise any over representation in any one variable during training. This method of cross-validation allows the full data dataset to be used for training the model without the issues of overfitting the model to the testing data.


#### *Outputs*
The output for each will produce a [confusion matrix] to evaluate the quality of each of the classification methods. This model will be trained to predict the final grade "G3" of a student enrolled in Portuguese, given the input values from the survey.


[8]: <http://leitang.net/papers/ency-cross-validation.pdf>
[confusion matrix]: <https://en.wikipedia.org/wiki/Confusion_matrix>
