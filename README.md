# BharatIntern_DataScience
Task1 - Titanic Classification

The "Titanic Classification" initiative is a machine learning endeavor that centers around the prediction of passenger survival outcomes on the iconic RMS Titanic. By leveraging the Titanic dataset, encompassing passenger particulars such as age, gender, ticket class, and a binary label indicating survival, this project's central objective is to develop a predictive model that can discern patterns and make forecasts regarding passenger survival.

Essential Components:

Utilization of Data:
  - The project leverages the well-established Titanic dataset, a widely recognized resource within the realm of machine learning, containing passenger information and indicators of survival.

Survival Classification: 
  - The core mission involves the development of a model with the capability to classify passengers into two distinct categories: those who experienced survival and those who did not.

Enhancement of Features: 
  - This phase frequently encompasses data preprocessing and feature engineering to uncover valuable insights within the dataset. It includes addressing missing data and creating new features.

Modeling Approach: 
  - Various machine learning algorithms, ranging from decision trees and random forests to logistic regression and others, may be utilized to create customized predictive models suited for this specific classification objective.

Assessment of Performance: 
  - The project assesses the effectiveness of the model by employing performance metrics such as accuracy, precision, recall, and F1 score to measure the model's competency in predicting survival outcomes.

The model's predictive capabilities can be adapted for various scenarios, from disaster preparedness to risk assessment, offering a tool that holds significance in diverse domains.Moreover, this project serves as an invaluable educational resource, allowing individuals to grasp the fundamental principles of machine learning and data analysis, particularly within the realm of classification tasks.



TASK 2 - Number Recognition

Importing Essential Libraries
- Import necessary libraries for the project, including 'os,' 'cv2' (for image processing), 'numpy' (for numerical computations), 'matplotlib' (for image visualization), and 'tensorflow' (for machine learning and deep learning tasks).

Loading and Preprocessing the MNIST Dataset**
- Load the MNIST dataset, which comprises images of handwritten digits paired with their labels.
- Normalize the pixel values of the images to fall within the 0 to 1 range, improving training efficiency.

Defining the Neural Network Model
- Create a Sequential model using TensorFlow/Keras.
- Flatten the 28x28 pixel images into a 1D array.
- Add two dense layers with ReLU activation functions for feature extraction.
- Append the output layer with 10 units and utilize softmax activation for digit classification.
- Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.

Training the Model
- Train the model by using the training data (x_train and y_train) over three epochs.
- The model acquires the ability to classify handwritten digits based on the training dataset.

Evaluating the Model
- Assess the performance of the trained model on the test data (x_test and y_test).
- Report the model's loss and accuracy on the test dataset.

Loading and Making Predictions on Desktop Images

- Define the path to the folder containing digit images on the desktop.
- List all files in the folder with the ".png" extension.
- Iterate through the image files, loading each image using OpenCV and extracting the first channel (assuming grayscale).
- Adjust the image colors if necessary.
- Utilize the trained model to predict the digit in the image.
- Display both the image and the predicted digit using matplotlib.
- Handle any exceptions or errors that may occur during the process.
