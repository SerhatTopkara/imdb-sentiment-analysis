# IMDb Movie Review Sentiment Analysis and Visualization

This project performs sentiment analysis on IMDb movie reviews using machine learning and natural language processing (NLP) techniques. It determines whether reviews are positive or negative and visualizes the results.

## Project Description

In this project:
- Data cleaning and preprocessing of IMDb movie review dataset
- Text vectorization using TF-IDF
- Training of machine learning models for sentiment analysis
- Visualization of results

## Features

- Data preprocessing and cleaning
- Multiple model comparison (Logistic Regression, Naive Bayes, Random Forest)
- Comprehensive visualization of results
- Interactive command-line interface
- Modular code structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SerhatTopkara/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Install required packages:
```bash
chmod +x run.sh
./run.sh
```
## For Windows Users:

You can start the project by running the following command:

```bat
run_Windows.bat

## Usage

Run the project:
```bash
./run.sh
```

Optional arguments:
```bash
./run.sh --skip-preprocessing  # Skip data preprocessing
./run.sh --skip-training       # Skip model training
./run.sh --skip-visualization  # Skip visualization
```

## Project Structure

- `src/`: Source code
  - `data_preprocessing.py`: Data cleaning and preprocessing functions
  - `model_training.py`: Model training and evaluation
  - `visualization.py`: Results visualization
  - `main.py`: Main program file
- `data/`: Dataset and processed data
- `models/`: Trained models
- `results/`: Visualizations and analysis results

## Results

Performance of different algorithms:
- Logistic Regression: Accuracy = ~82.25%
- Naive Bayes: Accuracy = ~80.75%
- Random Forest: Accuracy = ~79.50%

## Visualizations

Project results are visualized in the `results/` folder:
- Sentiment distribution pie chart
- Word clouds for most frequent words
- Model comparison bar chart
- Confusion matrices for each model

