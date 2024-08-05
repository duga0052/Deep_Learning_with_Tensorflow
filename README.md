Deep Learning with tensorflow
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

Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"