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

## Instructions to Run the Code

To run this code and replicate the experiments, follow these steps:

1. Clone the Repository:
      
   ```bash
   git clone https://github.com/gabriel-nds/CNN_for_Digit_Recognizer.git

2. Create and Activate a Virtual Environment:

    Create a virtual environment (recommended) and activate it. This helps isolate the        project's dependencies from your global Python environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows, use "venv\Scripts\activate"

3. Install Dependencies:

   Navigate to the project directory and install the required dependencies using the requirements.txt file:

   ```bash
   pip install -r requirements.txt

4. Run the Code:

   Open the digit.ipynb Jupyter Notebook in your preferred environment and execute the code cells. This notebook contains the code for training the model and evaluating its performance.

5. View Results:

   After executing the code cells in the digit.ipynb notebook, you'll see visualizations and printed results within the notebook itself. This includes insights about the model's predictions, a confusion matrix, and an analysis of incorrect predictions.

6. Experiment and Customize:
   
   Feel free to experiment with different hyperparameters, visualization options, and code components within the digit.ipynb notebook to further analyze the model and its performance.

## Model Performance on Kaggle Competition

Before we dive into the notebook's structure, let's celebrate our accomplishments. In the Digit Recognizer Kaggle competition, our model has demonstrated exceptional performance, ranking in the top 9% of participants: 

![64F9D1E5-5EB5-4F8D-B422-5AE9534FA87E_4_5005_c](https://github.com/gabriel-nds/CNN_for_Digit_Recognizer/assets/118403829/3a0061cc-918f-431c-a900-8110c2b69520)

On Kaggle we got a score of **99.49%**!

![3FEA9AE4-B073-408A-9957-554C8FF4B9A9](https://github.com/gabriel-nds/CNN_for_Digit_Recognizer/assets/118403829/78004796-b207-41cd-9751-3a7e87fd8f96)

This is a testament to the effectiveness of the techniques and strategies we'll explore in this notebook.

### Final Epoch Log
In the last epoch of training, our model achieved the following results:

- Training Loss: 0.0126
- Training Accuracy: 99.60%
- Validation Loss: 0.0134
- Validation Accuracy: 99.60%

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
11. Evaluating the Model: We assess the model's performance by visually examining a sample of validation set predictions and comparing them against real labels. Additionally, we compute a confusion matrix to gain insights into classification trends and identify areas of improvement. The percentage of incorrect predictions is calculated for overall assessment.
12. Create Submission: Finally, we use the trained model to make predictions on the test dataset and create a submission file for the competition.

---

This project demonstrates how to build and train a convolutional neural network for digit recognition using the MNIST dataset. By following the steps outlined above, you can replicate the process and explore different variations to improve model performance.






By following along with this notebook, you'll gain a better understanding of how to implement and fine-tune a CNN for image classification tasks. Feel free to experiment with different architectures, hyperparameters, and techniques to improve the model's accuracy.

Let's get started!
