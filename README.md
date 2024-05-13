# Toxic Comment Classification

## Overview

Toxic Comment Classification is a project aimed at building a machine learning model to classify toxic comments from online conversations. The goal is to develop a model that can automatically identify and flag toxic or offensive comments, helping to maintain healthy and respectful discussions in online communities.

This project utilizes natural language processing (NLP) techniques and deep learning algorithms to analyze text data and classify comments into different categories of toxicity, such as toxic, severe toxic, obscene, threat, insult, and identity hate.

## Dataset

The dataset used in this project is sourced from the [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). It contains a large number of YouTube comments which have been labeled by human raters for toxic behavior.

The dataset contains the following columns:
- `id`: Unique identifier for each comment
- `comment_text`: The text of the comment
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Binary labels indicating the presence of different types of toxicity in the comment


## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/N3RaJ/Toxic-Comment-Classification.git

2. **Install Dependencies**
   ```bash
   cd Toxic-Comment-Classification
   pip install -r requirements.txt

3. **Download Dataset**:
   Download the Kaggle Toxic Comment Classification Challenge dataset.
   Place the train.csv file in the data/ directory.

4. Explore Data and Train Model:

Explore the provided Jupyter notebooks in the notebooks/ directory for data exploration and model development.

5. **Evaluate Model**:
   Evaluate the trained model using various metrics.
   Tune hyperparameters as needed.

6. **Make Predictions**:
   
## Technologies Used
Python
TensorFlow
Keras
Pandas
Scikit-learn
Jupyter Notebook

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

