# MissedMedicalAppointmentPredictionUsingMachineLearning_PythonLanguage_FinalYearProjectUMS (This is to check the accuracy of the machine learning prediction in Jupyter_Anaconda)

 In:  import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import matplotlib.colors as colors
      from sklearn.utils import resample
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import scale
      from sklearn.svm import SVC
      from sklearn.model_selection import GridSearchCV
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import plot_confusion_matrix
      from sklearn.decomposition import PCA
      
 In:  df = pd.read_csv('No_show_issue.csv',
                      header=1)
 In:  df.head()
 Out: PatientId	AppointmentID	Gender	ScheduledDay	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	2.990000e+13	5642903	2	2016-04-29T18:38:08Z	2016-04-29T00:00:00Z	62	JARDIM DA PENHA	2	1	2	2	2	2	2
      1	5.590000e+14	5642503	1	2016-04-29T16:08:27Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	2	2	2	2	2	2
      2	4.260000e+12	5642549	2	2016-04-29T16:19:04Z	2016-04-29T00:00:00Z	62	MATA DA PRAIA	2	2	2	2	2	2	2
      3	8.680000e+11	5642828	2	2016-04-29T17:29:31Z	2016-04-29T00:00:00Z	8	PONTAL DE CAMBURI	2	2	2	2	2	2	2
      4	8.840000e+12	5642494	2	2016-04-29T16:07:23Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	1	1	2	2	2	2
      
 In:  df.drop('PatientId', axis=1, inplace=True) #set axis=0 to remove rows, axis=1 to remove columns
      df.head()
 Out: AppointmentID	Gender	ScheduledDay	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	5642903	2	2016-04-29T18:38:08Z	2016-04-29T00:00:00Z	62	JARDIM DA PENHA	2	1	2	2	2	2	2
      1	5642503	1	2016-04-29T16:08:27Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	2	2	2	2	2	2
      2	5642549	2	2016-04-29T16:19:04Z	2016-04-29T00:00:00Z	62	MATA DA PRAIA	2	2	2	2	2	2	2
      3	5642828	2	2016-04-29T17:29:31Z	2016-04-29T00:00:00Z	8	PONTAL DE CAMBURI	2	2	2	2	2	2	2
      4	5642494	2	2016-04-29T16:07:23Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	1	1	2	2	2	2
      
In:   df.drop('AppointmentID', axis=1, inplace=True)
      df.head()
Out:  Gender	ScheduledDay	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	2	2016-04-29T18:38:08Z	2016-04-29T00:00:00Z	62	JARDIM DA PENHA	2	1	2	2	2	2	2
      1	1	2016-04-29T16:08:27Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	2	2	2	2	2	2
      2	2	2016-04-29T16:19:04Z	2016-04-29T00:00:00Z	62	MATA DA PRAIA	2	2	2	2	2	2	2
      3	2	2016-04-29T17:29:31Z	2016-04-29T00:00:00Z	8	PONTAL DE CAMBURI	2	2	2	2	2	2	2
      4	2	2016-04-29T16:07:23Z	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	1	1	2	2	2	2
      
In:   df.drop('ScheduledDay', axis=1, inplace=True)
      df.head()
Out:  Gender	AppointmentDay	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	2	2016-04-29T00:00:00Z	62	JARDIM DA PENHA	2	1	2	2	2	2	2
      1	1	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	2	2	2	2	2	2
      2	2	2016-04-29T00:00:00Z	62	MATA DA PRAIA	2	2	2	2	2	2	2
      3	2	2016-04-29T00:00:00Z	8	PONTAL DE CAMBURI	2	2	2	2	2	2	2
      4	2	2016-04-29T00:00:00Z	56	JARDIM DA PENHA	2	1	1	2	2	2	2
      
In:   df.drop('AppointmentDay', axis=1, inplace=True)
      df.head()
Out:  Gender	Age	Neighbourhood	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	2	62	JARDIM DA PENHA	2	1	2	2	2	2	2
      1	1	56	JARDIM DA PENHA	2	2	2	2	2	2	2
      2	2	62	MATA DA PRAIA	2	2	2	2	2	2	2
      3	2	8	PONTAL DE CAMBURI	2	2	2	2	2	2	2
      4	2	56	JARDIM DA PENHA	2	1	1	2	2	2	2
      
In:   df.drop('Neighbourhood', axis=1, inplace=True)
      df.head()
Out:  Gender	Age	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	No-show
      0	2	62	2	1	2	2	2	2	2
      1	1	56	2	2	2	2	2	2	2
      2	2	62	2	2	2	2	2	2	2
      3	2	8	2	2	2	2	2	2	2
      4	2	56	2	1	1	2	2	2	2
      
In:   df.dtypes #identify missing data
Out:  Gender          int64
      Age             int64
      Scholarship     int64
      Hipertension    int64
      Diabetes        int64
      Alcoholism      int64
      Handcap         int64
      SMS_received    int64
      No-show         int64
      dtype: object
      
In:   df['Gender'].unique()
Out:  array([2, 1], dtype=int64)

In:   df['Age'].unique()
Out:  array([62, 56,  8, 76, 23, 39, 21, 19, 30, 29, 22, 28, 54, 15, 50, 40, 46,
              4, 13, 65, 45, 51, 32, 12, 61, 38, 79, 18, 63, 64, 85, 59, 55, 71,
             49, 78, 31, 58, 27,  6,  2, 11,  7,  0,  3,  1, 69, 68, 60, 67, 36,
             10, 35, 20, 26, 34, 33, 16, 42,  5, 47, 17, 41, 44, 37, 24, 66, 77,
             81, 70, 53, 75, 73, 52, 74, 43, 89, 57, 14,  9, 48, 83, 72, 25, 80,
             87, 88, 84, 82, 90, 94, 86, 91, 98], dtype=int64)
             
In:   df['Scholarship'].unique()
Out:  array([2, 1, 0], dtype=int64)

In:   df['Hipertension'].unique()
Out:  array([1, 2], dtype=int64)

In:   df['Diabetes'].unique()
Out:  array([2, 1], dtype=int64)

In:   df['Alcoholism'].unique()
Out:  array([2, 1], dtype=int64)

In:   df['Handcap'].unique()
Out:  array([2, 1], dtype=int64)

In:   df['SMS_received'].unique()
Out:  array([2, 1], dtype=int64)

In:   df['No-show'].unique()
Out:  array([2, 1], dtype=int64)

In:   #dealing with missing data
      len(df.loc[(df['Age'] == 0) | (df['Scholarship'] == 0)])
Out:  38

In:   len(df)
Out   998

In:   df_no_missing = df.loc[(df['Age'] !=0) & (df['Scholarship'] !=0)]
In:   len(df_no_missing)
Out:  960

In:   df_no_missing['Age'].unique()
Out:  array([62, 56,  8, 76, 23, 39, 21, 19, 30, 29, 22, 28, 54, 15, 50, 40, 46,
              4, 13, 65, 45, 51, 32, 12, 61, 38, 79, 18, 63, 64, 85, 59, 55, 71,
             49, 78, 31, 58, 27,  6,  2, 11,  7,  3,  1, 69, 68, 60, 67, 36, 10,
             35, 20, 26, 34, 33, 16, 42,  5, 47, 17, 41, 44, 37, 24, 66, 77, 81,
             70, 53, 75, 73, 52, 74, 43, 89, 57, 14,  9, 48, 83, 72, 25, 80, 87,
             88, 84, 82, 90, 94, 86, 91, 98], dtype=int64)
             
In:   df_no_missing['Scholarship'].unique()
Out:  array([2, 1], dtype=int64)

In:   #downsample the data
      len(df_no_missing)
Out:  960

In:   df_missed = df_no_missing[df_no_missing['No-show'] == 1]
      df_not_missed = df_no_missing[df_no_missing['No-show'] == 2]
In:   df_missed_downsampled = resample(df_missed,
                                      replace=False,
                                      n_samples=100,
                                      random_state=42)
      len(df_missed_downsampled)
Out:  100

In:   df_not_missed_downsampled = resample(df_not_missed,
                                          replace=False,
                                          n_samples=100,
                                          random_state=42)
      len(df_not_missed_downsampled)
Out:  100

In:   df_downsample = pd.concat([df_missed_downsampled, df_not_missed_downsampled])
      len(df_downsample)
Out:  200

In:   #Format data: split data into dependent and independent variables
      x = df_downsample.drop('No-show', axis=1).copy() 
      x.head()
Out:  Gender	Age	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received
      631	2	27	2	2	2	2	2	2
      669	2	33	2	2	2	2	1	2
      852	2	23	2	2	2	2	2	2
      298	2	22	2	2	2	2	2	2
      555	2	28	2	2	2	2	2	1
      
In:   y = df_downsample['No-show'].copy()
      y.head()
Out:  631    1
      669    1
      852    1
      298    1
      555    1
      Name: No-show, dtype: int64
      
In:   #Format the data: one-Hot Encoding
      pd.get_dummies(x, columns=['Gender']).head()
Out:  Age	Scholarship	Hipertension	Diabetes	Alcoholism	Handcap	SMS_received	Gender_1	Gender_2
      631	27	2	2	2	2	2	2	0	1
      669	33	2	2	2	2	1	2	0	1
      852	23	2	2	2	2	2	2	0	1
      298	22	2	2	2	2	2	2	0	1
      555	28	2	2	2	2	2	1	0	1
      
In:   x_encoded = pd.get_dummies(x, columns=['Gender',
                                            'Scholarship',
                                            'Hipertension',
                                            'Diabetes',
                                            'Alcoholism',
                                            'Handcap',
                                            'SMS_received'])
      x_encoded.head()
Out:  Age	Gender_1	Gender_2	Scholarship_1	Scholarship_2	Hipertension_1	Hipertension_2	Diabetes_1	Diabetes_2	Alcoholism_1	Alcoholism_2	Handcap_1	Handcap_2	SMS_received_1	SMS_received_2
      631	27	0	1	0	1	0	1	0	1	0	1	0	1	0	1
      669	33	0	1	0	1	0	1	0	1	0	1	1	0	0	1
      852	23	0	1	0	1	0	1	0	1	0	1	0	1	0	1
      298	22	0	1	0	1	0	1	0	1	0	1	0	1	0	1
      555	28	0	1	0	1	0	1	0	1	0	1	0	1	1	0
      
In:   x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, random_state=42)
      x_train_scaled = scale(x_train)
      x_test_scaled = scale(x_test)
In:   #Build a preliminary SVM
      clf_svm = SVC(random_state=42)
      clf_svm.fit(x_train_scaled, y_train)
Out:  SVC(random_state=42)

In:   plot_confusion_matrix(clf_svm,
                           x_test_scaled,
                           y_test,
                           values_format='d',
                           display_labels=["Missed", 'Not Missed'])
Out:  <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x241f198b910>

In:   #Optimize Parameters with cross validation & GridSearchCV()
      param_grid = [
          {'C': [0.5, 1, 10, 100],  #value for C must be > 0
           'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
           'kernel': ['rbf']},
      ]
      #including C=1  and gamma='scale' as possible choices since they are default values.

      optimal_params = GridSearchCV(
              SVC(),
              param_grid,
              cv=5,
              scoring='accuracy',
              verbose=0  #if u want to see what Grid Search is doing, set verbose=2
          )

      optimal_params.fit(x_train_scaled, y_train)
      print(optimal_params.best_params_)
Out:  {'C': 0.5, 'gamma': 'scale', 'kernel': 'rbf'}

In:   #Building, evaluating, drawing, and interpreting final SVM
      clf_svm = SVC(random_state=42, C=0.5, gamma='scale')
      clf_svm.fit(x_train_scaled, y_train)
Out:  SVC(C=0.5, random_state=42)

In:   plot_confusion_matrix(clf_svm,
                           x_test_scaled,
                           y_test,
                           values_format='d',
                           display_labels=["Missed", "Not Missed"])
Out:  <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x241f2167220>

In:   len(df_downsample.columns)
Out:  9

In:   pca = PCA() #By default, PCA() centers the data, but does not scale it
      x_train_pca = pca.fit_transform(x_train_scaled)

      per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
      labels = [str(x) for x in range(1, len(per_var)+1)]

      plt.bar(x=range(1,len(per_var)+1), height=per_var)
      plt.tick_params(
          axis='x',      #change apply to the x-axis
          which='both',  #both major and minor ticks are affected
          bottom=False,  #ticks along the bottom edge are off
          top=False,     #ticks along the top edge are off
          labelbottom=False)   #labels along the bottom edge are off
      plt.ylabel('Percentage of Explained Variance')
      plt.xlabel('Principle Components')
      plt.title('Scree Plot')
      plt.show()

 In:  train_pc1_coords = x_train_pca[:, 0]
      train_pc2_coords = x_train_pca[:, 1]

      #pc1 contains x-axis coordinates of the data after PCA
      #pc2 contains y-axis cordinates of data after PCA

      #Now center and scale the PCs...
      pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

      #Now we optimize the SVM fit to the x and y-axis coordinates of the data after PCA dimension reduction
      param_grid = [
          {'C': [1, 10, 100, 1000],
          'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
          'kernel': ['rbf']},
      ]

      optimal_params = GridSearchCV(
              SVC(),
              param_grid,
              cv=5,
              scoring='accuracy',
              verbose=0
          )

      optimal_params.fit(pca_train_scaled, y_train)
      print(optimal_params.best_params_)
Out:  {'C': 1, 'gamma': 1, 'kernel': 'rbf'}

In:   clf_svm = SVC(random_state=42, C=1, gamma=1)
      clf_svm.fit(pca_train_scaled, y_train)

      #Transform the test dataset with the PCA..
      x_test_pca = pca.transform(x_train_scaled)
      #x_test_pca = pca.transform(x_test_scaled)
      test_pc1_coords = x_test_pca[:, 0]
      test_pc2_coords = x_test_pca[:, 1]

      #Now create a matrix of points that we use to show the decision regions
      #The matrix will be a little bit larger that transformed PCA points so that we can plot all of
      #the PCA points on it without them being on the edge
      x_min = test_pc1_coords.min() - 1
      x_max = test_pc1_coords.max() + 1

      y_min = test_pc2_coords.min() - 1
      y_max = test_pc2_coords.max() + 1

      xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                          np.arange(start=y_min, stop=y_max, step=0.1))

      #Now we will classify every point in that matrix with SVM point in that
      #matrix with SVM. Points on one side of the classification boundary will get 0, and points on the other
      #side will get 1.
      z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
      #Right now, z is just along array of lots of 0s and 1s, which
      #reflect how each point in the mesh was classified.
      #We use reshape() so that each classification (0 or 1) corresponds
      #to a specific point in the matrix.
      z = z.reshape(xx.shape)

      fig, ax = plt.subplots(figsize=(10,10))
      #Now we will use contour() to draw a filled contour plot using the matrix values and classifications.
      #The contour will be filled according to the predicted classifications (0s and 1s) in z
      ax.contourf(xx, yy, z, alpha=0.1)

      #Now create custom colours for the actual data points
      cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
      #Now draw the actual data points - these will be colored by their known (not predicted) classifications
      scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                      cmap=cmap,
                      s=100,
                      edgecolors='k', #'k'=black
                      alpha=0.7)

      #Now create a legend
      legend = ax.legend(scatter.legend_elements()[0],
                         scatter.legend_elements()[1],
                            loc="upper right")
      legend.get_texts()[0].set_text("Yes Missed")
      legend.get_texts()[1].set_text("Not Missed")

      #Now add axis labels and titles
      ax.set_ylabel('PC2')
      ax.set_xlabel('PC1')
      ax.set_title('Decision surface using the PCA transformed/projected features')
      #plt.savefig('svm_default.png')
      plt.show()
