# Football Match Prediction using Random Forest

## Overview
This project demonstrates the use of a Random Forest Classifier to predict the outcomes of football matches based on historical data. The model is trained using match data, where rolling averages of match statistics such as goals and shots are used to generate features for the prediction model.

## Features
- Preprocessing and feature engineering using Pandas.
- Rolling averages of key statistics for teams.
- Random Forest Classifier model built with Scikit-learn.
- Evaluation of the model using accuracy and precision scores.

## Technologies
- **Python**
- **Pandas**
- **Scikit-learn**
- **Random Forest**
- **Data Preprocessing**
- **Feature Engineering**

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/football-match-prediction.git
   cd football-match-prediction
   ```

2. **Create and activate a virtual environment**:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script**:
   ```bash
   python main.py
   ```

## License
This project is licensed under the MIT License.
