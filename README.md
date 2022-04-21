# [Classifying “Essential Worker” Trips in NYC During the COVID-19 Lockdown](https://github.com/JeremyR327/EssentialWorkerRides_ML_Model/files/8531498/MLC_FinalProjectPaper.pdf)
Machine Learning for Cities Project

This github repository contains all relavant data preprocessing and modeling techniques used for this machine learning project. For the full PDF report click the above link.

## Abstract


In May 2020, the Metropolitan Transportation Authority (MTA) suspended all overnight subway service, between 1 AM and 5 AM through May 2021, impacting these workers who were often commuting overnight by public transportation prior to lockdown. As a result, the city offered subsidized for-hire vehicle (FHV) rides to essential workers during subway closure. This program, known as the Essential Connector program, only ran from May 6 to August 31, 2020 before the city established expanded overnight bus service to replace it. This setup offers a unique research opportunity: during the three months that the program existed, overnight rides were subsidized for essential workers, which we assume to be the majority of overnight riders in the middle of a pandemic.

Our goal was to exploit this natural experiment to classify overnight FHV rides during this time period as “essential worker rides” in order to model the 
classification of essential rides based on Census data and other transportation information. We hypothesized that some basic demographic information of 
pickup and dropoff zones, such as commuting patterns, employment sector, and household income, combined with subway data and ride characteristics, 
such as day of week, duration, and FHV provider (i.e. Uber or Lyft), could be used to classify a ride as overnight, which by proxy represents an essential 
worker ride. Our results show that overnight “essential worker” rides can be predicted with around 70% accuracy with 55% recall using these input variables. 
Furthermore, our most accurate model is computationally efficient and quite interpretable, making the results applicable to public sector use.

## Methodology

### Feature Engineering
First we needed to create our target variable: whether or not a given FHV trip started
between the hours of 1 and 5 AM. Using the datetime information provided, we extracted the time of the ride pickup and created a binary classifier to classify each ride as "overnight” or “non-overnight”.
We also created a ‘day of week’ variable for each ride to account for periodicity throughout the week. These seven categorical values (Monday through Sunday) were then encoded as dummy variables.
Next, we mapped another binary variable from the geocoded subway information, delineating whether a given taxi zone contains any subway stations, regardless of ridership. This is labeled as “Subway_x” and “Subway_y”. Since many outer-borough areas do not have subway access, this was a better approach.

##### ACS Principal Component Analysis

Considering the millions of observations, we quickly ran into computational concerns when mapping the nine ACS variables to both the Pickup Location and Dropoff Location of each ride. As an alternative, we turned to Principal Component Analysis, using SKLearn to perform a Linear PCA transformation on the taxi zone-level ACS data. This serves two purposes: the first is to remove any multicollinearity that likely exists in social measurements, while the second is to limit dimensionality.
 
 We scaled the data to account for the various distributions of the ACS variables. Then, we decomposed it into a single principal component, so that each taxi zone has a corresponding value for it’s ACS information. These values were then mapped to each pickup and dropoff Location of the ride level dataset, which we will refer to in the model as variables “PCA_x” and “PCA_y.”

##### Pickup and Dropoff Clustering

We also decided to remove the actual pickup and dropoff locations from the model, since we’re more interested in modeling the characteristics of rides rather than the geographical attributes of the rides themselves. A classifier could accurately predict overnight rides based on location, but that wouldn’t provide much insight for our topic (and would raise serious spatial autocorrelation concerns).
Instead, we created mobility clusters with the taxi zones, using total ride counts as well other summed ridership variables (such as rides by day of week and overnight rides). We applied KMeans clustering to this mobility count data and split them into 5 clusters, as suggested by the Silhouette Score metric. We again mapped these cluster assignments to the Pickup and Dropoff Location of each ride, labeled as “PU_Cluster” and “DO_Cluster”. These variables were then given dummy assignments.

### Modelling

##### Data Sampling

At first, modeling efforts yielded accuracy of over 90% for nearly all models. However, we quickly discovered this was because our target classes were extremely unbalanced, with “overnight” rides comprising just 7% of the randomly sampled dataset. We used imlearn’s RandomUnderSampler module to create a 55/45 split between the two target classes. This left us with just over 3 million evenly balanced entries. The resulting model selection was performed using samples with this target class split.
##### Model Selection

For this project, we are interested in both binary and probabilistic classification for each ride and predicted probability scores were included for all models. We first trained and tested the logistic regression again on the balanced dataset. As expected, the accuracy dropped dramatically, correctly predicting just 59% of the training and validation data. Despite its poor accuracy, it retained a reasonable ROC AUC sore and its variable coefficients provided interesting insight. According to the logistic model, “Duration” and the “Sunday” binary had the biggest classification impact, followed by the other weekday variables and the “ACS PCA“ features.

![ValidationLogisticRegressionCoefficients](https://user-images.githubusercontent.com/17898669/164475672-2b3924b4-5e06-45b2-b4f5-3c8390594e45.png)

Our “in sample” accuracy improved dramatically with a decision tree classifier, so we decided to fit the data using a random forest model as well as Python’s XGBoost gradient booster. Both the random forest and XGBoost model provided improved results, but a tuned random forest model provided the best output while maintaining interpretability. We used several iterations of GridSearchCV and RandomizedSearchCV for hyperparameter tuning over the validation set, landing on the following combination: n_estimators = 100, max_depth = 30, min_samples_split = 5, max_features = 'sqrt ', min_samples_leaf = 4 .
The selected model was then tested again on the validation set, yielding the highest results yet, including an “out of sample” accuracy of 0.643 and a ROC AUC score of 0.693.
As a quick aside, a support vector machine classifier would have been well-suited for this type of classification, since it’s unclear if our decision boundary is linear enough to map with a standard logistic regression. However, given the extremely high dimensionality of the data, we were not able to successfully fit or model an SVM instance.
##### Model Performance on Test Set

As mentioned before, the original dataset of nearly 32 million rides was randomly split to create a more manageable engineering space. As an added bonus, it also gave us a robust isolated test dataset of unseen observations to use a final “out of sample” test for evaluating our model performance.
We used this final test set (with about 3 million entries) on our fitted random forest classifier and, for an added layer of difficulty, we left the set unbalanced, so that it retained a real-world ratio of non-overnight to overnight rides.
Even with an unbalanced sample, our model produced better results than those generated from the validation sample.
   
Despite a very low precision score (our classifier predicted substantially more overnight rides than actual ones), this model performed decently well on recall and accuracy while its ROC represents a favorable balance.
The model also provided information on it’s most important features, with the intensity of each generally mirroring the coefficient magnitudes seen in the logistic classifier.

![FINALRandomForestFeatureImportances](https://user-images.githubusercontent.com/17898669/164475943-758f0c8c-bb4a-459f-8d9e-23a4736db094.png)

Our random forest model was dominated by the “Duration” variable, with an impurity metric of just over 0.4, more than twice that of any other variable in the ensemble model. The next two most prominent features are the ACS PCA values, with scores of 0.2 for both pickup (x) and dropoff (y) locations. Conceptually, this is incredibly important, since using demographic information about ride location for modeling is an integral part of this project.
The probability and classification for each test ride was mapped to the full test dataset for further analysis of the model.

## Results & Discussion
Despite the computational and methodological difficulties presented by our large and imbalanced dataset, our final random forest classifier showed promising results in support of our initial hypothesis. Using basic ride characteristics such as the day of week and trip duration, coupled with publicly available demographic data, we were able to predict overnight, assumed essential, rides with 72% out-of-sample accuracy. Even more promising, we were able to reach this level of accuracy with a relatively computationally inexpensive and interpretable model. Computational efficiency and interpretability were both priorities for this project, since the audience for its findings are likely government agencies such as the MTA and TLC.
The random forest classifier allows us to look beyond basic accuracy and into feature importance to infer what may be influencing classification most. By far the most influential feature is trip duration, calculated in minutes, which has a negative coefficient in our Logistic Regression model. The fact that trips classified as overnight are shorter could be a result of reduced traffic, or it could be that trips overnight are shorter in distance. The Essential Connector service was aimed at commuters who needed to travel farther to get to a bus stop, or required a longer bus commute, but not necessarily riders who needed to travel long distances. It’s possible that their actual commutes by car would have been quite quick, especially without traffic. They may also have taken Essential Connector rides to nearby, but not walking distance, bus stops or other forms of mixed mobility to shorten commutes, which would also shorten FHV trip duration.

![RideDurationDistribution](https://user-images.githubusercontent.com/17898669/164476371-1b6ea5e7-440e-449b-8b81-736eb4686c43.png)
 
 “Day of week” was also an important feature, with the dummy variable for Sundays a substantial predictive impact across all models. It’s important to note that Sunday overnight rides are technically considered Saturday night, while Sunday daytime rides would have occurred on Sunday during the day. It is possible that our model is influenced by some Sunday overnight riders that are not taking FHVs for essential purposes. Saturday nights and Sunday mornings would have been the most common type of overnight ride pre-pandemic, when nightlife was still at full capacity in New York. It’s difficult to know whether people were still going between each other's residences and possibly public places overnight last summer.
Finally, the ACS principal components for both pickup and dropoff locations had the second and third highest feature importance in the test set’s model. We expected this, because the ACS variables chosen were closely related to industry and to commuting patterns. The variable for the number of people working in healthcare, food service, maintenance, and retail sales, is especially useful as a proxy for where essential workers live. As Figure 10 shows, the locations with high predicted overnight ride counts are similar to the areas with high essential worker residency counts.

![PredictedRidesVSEssentialWorkers](https://user-images.githubusercontent.com/17898669/164476453-947d4a87-04b2-457d-923f-bc6c709fcf4c.png)


Although accuracy and recall for our classifier are relatively high, the model precision is only 15%. This manifests as the model predicting over three times as many overnight rides as there actually were. This is likely due to the unbalanced nature of the test set, and represents a major limitation of our model. The precision was much higher for the undersampled validation set, at around 61%, supporting this assumption. Another possibility for the poor precision score could be that some of the variables, specifically our ACS principal components, are also predicting daytime essential rides as overnight. Although these rides did not happen overnight,
they would have had similar ACS principal component values and possibly other similar features to overnight essential rides. It would make sense that essential workers took FHV rides more than normal during the pandemic, even when the subway was operating, in order to avoid public transportation crowds.

![Screen Shot 2022-04-21 at 10 09 59 AM](https://user-images.githubusercontent.com/17898669/164476694-7b7b52e4-49e6-44f5-96d7-bc19b3a5f3f3.png)

Finally, as a sanity check, it is useful to compare our map of overnight rides with that of Essential Connector pickups created by the Rudin Center’s report, as shown in Figure 12. East New York and Crown Heights are the two most frequented overnight locations in both the true dataset and our model, and they are listed in the Rudin report as having some of the highest Essential Connector ridership. These areas are also places where our model was most prone to over-classifying overnight rides, shown in red in Figure 11. This overlap supports our assumption that overnight rides would have been primarily “essential,” and by proxy the assumption that our model might be predicting essential rides, not just those that happen to occur overnight.

## Conclusion
We can state with relative confidence that our classification model accurately predicts whether a ride occurred overnight. We also have some exploratory evidence supporting our assumption that most overnight rides were essential. This means that our model is somewhat accurately predicting essential rides based on basic ride and neighborhood characteristics. The value of this finding is widespread and applicable beyond the highly unusual setting in which our model was trained. If anything, the global pandemic has demonstrated the vital importance of essential workers to urban systems, and with that the importance of sustainable and affordable transportation infrastructure for those workers. Even after New York returns to relative normalcy
  
and the MTA reinstates overnight service on May 17, 2021, many essential workers will still have the same transportation troubles as they did in summer of 2020, many of them overnight. Being able to determine where and how essential workers are commuting gives New York City a unique opportunity to prioritize affordable transportation for those populations beyond COVID-19. With actual information on the use of the Essential Connector service or more detailed trip information, it’s possible for the MTA and TLC to gain a true understanding of essential workers’ transportation needs.

