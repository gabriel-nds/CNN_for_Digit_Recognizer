![Python](https://img.shields.io/badge/Python-3.10.12-pink)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11.0-green)
![Keras](https://img.shields.io/badge/Keras-2.11-blue)

# DeepDigits: Handwritten Digit Recognition with Convolutional Neural Networks

## About the Project

Welcome to the Digit Recognizer Project! This Jupyter Notebook contains the code and steps for building a convolutional neural network (CNN) to recognize handwritten digits. The goal of this project is to demonstrate the process of creating, training, and evaluating a CNN using the popular MNIST dataset.

The MNIST dataset is a widely recognized benchmark in the fields of machine learning and computer vision. It comprises a collection of grayscale images, each representing a handwritten digit ranging from 0 to 9. These images are 28x28 pixels in size and have been extensively used to develop and test various image processing and classification algorithms. The dataset was derived from a larger collection of handwritten digits provided by participants of the US Census Bureau and the National Institute of Standards and Technology (NIST).

Here's a glimpse of what the MNIST dataset contains:

![image](https://github.com/gabriel-nds/CNN_for_Digit_Recognizer/assets/118403829/0cf45853-b562-46b4-830b-86d10d810d18)

In this project, we leverage the MNIST dataset to train our CNN to accurately recognize and classify these handwritten digits. Through this project, we'll guide you through the entire pipeline, from loading and preprocessing the dataset to training and evaluating the neural network. The final result will be a trained model that can identify handwritten digits with impressive accuracy. Whether you're new to neural networks or looking to strengthen your skills, this project will provide valuable insights into image classification and CNNs. Let's dive in and explore the fascinating world of digit recognition!

## Model Performance on Kaggle Competition

Before we dive into the notebook's structure, let's celebrate our accomplishments. In the Digit Recognizer Kaggle competition, our model has demonstrated exceptional performance, ranking in the top 11% of participants: 

![BF52755D-091B-4B75-A64E-18566E06BAC7_4_5005_c](https://github.com/gabriel-nds/CNN_for_Digit_Recognizer/assets/118403829/c9611efa-4131-49b3-ba8f-b5843ef630ed)

On Kaggle we got a score of **99.43%**!

![2324EF29-9654-4CA6-ABC7-B7F588B5428F_4_5005_c](https://github.com/gabriel-nds/CNN_for_Digit_Recognizer/assets/118403829/431ab300-2ae5-445f-b545-2f73c264390f)

This is a testament to the effectiveness of the techniques and strategies we'll explore in this notebook.

### Final Epoch Log
In the last epoch of training, our model achieved the following results:

Training Loss: 0.0186
Training Accuracy: 99.41%
Validation Loss: 0.0167
Validation Accuracy: 99.51%

These statistics reflect the robustness of our convolutional neural network (CNN) architecture and the effectiveness of the training process. As we proceed through the notebook, you'll gain insights into how we built and fine-tuned this model to achieve such remarkable accuracy.

Let's dive into the details of our Digit Recognizer project and learn how we accomplished these results!

## Notebook Structure

This notebook is organized into sections, each focusing on a specific aspect of the project. Here's a quick overview of what to expect in each section:

1. Import Necessary Libraries: We begin by importing the required libraries for data handling, model creation, and visualization.
2. Load Data: Next, we load the MNIST dataset using the tensorflow.keras.datasets module. This dataset comprises images of handwritten digits along with their labels.
3. Prepare Data for Training: Before training the model, we need to preprocess the data. This involves reshaping the images and normalizing pixel values.
4. Split Data into Training and Validation Sets: We split the dataset into training and validation sets using the train_test_split function from sklearn.model_selection.
5. Define the Neural Network Architecture: The architecture of the neural network is defined using the Keras Sequential model. It includes convolutional layers, max-pooling layers, dropout layers, and fully connected layers.
6. Compile the Model: The model is compiled with an optimizer, loss function, and evaluation metric using the compile method.
7. Data Augmentation: To prevent overfitting, we apply data augmentation techniques using the ImageDataGenerator from Keras. This helps the model generalize better to new data.
8. Learning Rate Annealing with ReduceLROnPlateau: We implement learning rate annealing using the ReduceLROnPlateau callback. The learning rate is adjusted based on validation loss and patience parameters.
9. Train the Model: The model is trained using the training data and validated using the validation set. The training process is monitored and improved using the callbacks we defined earlier.
10. Visualize Training and Validation Metrics: We visualize the training and validation accuracy and loss using matplotlib.
11. Make Predictions and Create Submission: Finally, we use the trained model to make predictions on the test dataset and create a submission file for the competition.

---

This project demonstrates how to build and train a convolutional neural network for digit recognition using the MNIST dataset. By following the steps outlined above, you can replicate the process and explore different variations to improve model performance.






By following along with this notebook, you'll gain a better understanding of how to implement and fine-tune a CNN for image classification tasks. Feel free to experiment with different architectures, hyperparameters, and techniques to improve the model's accuracy.

Let's get started!
