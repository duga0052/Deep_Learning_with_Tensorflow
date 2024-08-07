Deep Learning with tensorflow

Purpose:

This project aims to predict employee attrition using machine learning techniques. The dataset includes various features related to employees such as age, income, job satisfaction, and more. The project involves data loading, preprocessing, model training, evaluation, and visualization.

-------------------

How to Run This Code:

 - Ensure you have Python installed on your system along with the required packages.
 - Place employee_attrition.csv in the directory where the script is located.
 - Run the main.py script in a Python environment.

 -------------

Dependencies"

The following libraries are required:

pandas: For data manipulation and analysis
numpy: For numerical operations
matplotlib: For plotting graphs
seaborn: For data visualization
scikit-learn: For machine learning algorithms and evaluation metrics
tensorflow: For building and training neural network models
logging: For logging errors and information

--------------

Ensure they are installed using pip:

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

-------------


Project Structure

├── data/
│   └── employee_attrition.csv      # The dataset file
├── src/
│   ├── __init__.py                 # Makes src a package
│   ├── data/
│   │   ├── __init__.py             # Makes data a package
│   │   └── data_processing.py      # Module for loading and preprocessing data
│   ├── model/
│   │   ├── __init__.py             # Makes model a package
│   │   └── improved_model.py       # Module for model creation, compilation, training, and evaluation
│   ├── visualization/
│   │   ├── __init__.py             # Makes visualization a package
│   │   └── visualization.py        # Module for data visualization
├── main.py                         # Main script to run the project
├── README.md                       # Project description and instructions
├── requirements.txt                # List of dependencies
└── .gitignore                      # Git ignore file

--------------

Detailed Steps
1. Data Loading
The dataset is loaded from a CSV file named employee_attrition.csv using the load_data function from data_processing.py. This function reads the data into a pandas DataFrame and performs initial preprocessing.
2. Data Preprocessing
Preprocess Data: The preprocess_data function in data_processing.py handles the data preprocessing steps including scaling numerical features and encoding categorical features.
3. Model Creation and Training
Create and Compile Model: The create_improved_model and compile_model functions in improved_model.py create and compile a neural network model.
Train Model: The train_model function trains the model using the training data.
4. Model Evaluation
Evaluate Model: The evaluate_model function in improved_model.py evaluates the model on the test data.
5. Visualization
Plot Training Curves: The plot_training_curves function in visualization.py plots the training curves for accuracy and loss.
Plot Confusion Matrix: The plot_confusion_matrix function plots the confusion matrix for the predictions.
Plot Learning Rate vs. Loss: The plot_learning_rate_vs_loss function plots the learning rate versus the loss.

-------------

Conclusion
This project demonstrates a complete workflow for predicting employee attrition using machine learning. It covers data preprocessing, model training, evaluation, and visualization. The models are evaluated using accuracy scores and confusion matrices to ensure robustness and reliability. The neural network model provides a comprehensive approach to understanding and predicting employee attrition based on various features.

--------------

Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"