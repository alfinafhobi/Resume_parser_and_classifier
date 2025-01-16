# Resume Parser and Classifier
This project is a Resume parsing and Classification System that categorizes resumes into predefined job roles using machine learning. 


## Overview
The **Resume Parser and Classifier** is a machine learning-based application designed to:

1. **Parse Resumes**: Extract structured data such as skills, experience, and personal information from resumes.
2. **Classify Resumes**: Categorize resumes into specific job roles (e.g., Data Scientist, Python Developer) using a trained model.

This project demonstrates the potential of natural language processing (NLP) and machine learning in automating HR and recruitment processes.


## Features

### Resume Parsing
- Extracts information like:
  - Personal Details
  - Skills
  - Education
  - Work Experience

### Resume Classification
- Categorizes resumes into predefined job roles such as:
  - Data Scientist
  - Python Developer
  - Web Designer
  - HR Specialist
  - DevOps Engineer

### Visualization
- Graphical representation of:
  - Distribution of job categories using bar charts and pie charts.


## Technologies Used

| Technology      | Purpose                     |
|------------------|-----------------------------|
| Python           | Backend Development         |
| Pandas           | Data Manipulation           |
| NumPy            | Numerical Operations        |
| scikit-learn     | Machine Learning            |
| matplotlib, seaborn | Data Visualization       |
| Regex            | Text Cleaning and Parsing   |
| spaCy, NLTK      | Natural Language Processing |


## Workflow

1. **Data Loading**: Load the dataset containing resumes and their corresponding categories.
2. **Data Preprocessing**:
   - Clean text data using regular expressions.
   - Convert categories to numerical labels.
3. **Feature Extraction**:
   - Use TF-IDF vectorization to transform resume text into numerical features.
4. **Model Training**:
   - Train a multi-class classifier using the k-Nearest Neighbors (KNN) algorithm.
5. **Evaluation**:
   - Split data into training and testing sets.
   - Evaluate the model's accuracy.
6. **Deployment**:
   - Save the trained model and vectorizer using `pickle` for reuse.
   - Test the model with live resume inputs.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/resume-parser-classifier.git
   cd resume-parser-classifier
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download spaCy models:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. (Optional) Install NLTK dependencies:
   ```bash
   python -m nltk.downloader all
   ```


## Usage

1. Run the script for training and classification:
   ```bash
   python main.py
   ```
2. Parse and classify a sample resume:
   ```python
   from resume_parser import resumeparse

   data = resumeparse.read_file("sample_resume.pdf")
   print(data)
   ```
3. Test live predictions:
   ```python
   import pickle

   clf = pickle.load(open('clf.pkl', 'rb'))
   tfidf = pickle.load(open('tfidf.pkl', 'rb'))

   input_resume = "Extracted resume text here"
   transformed_input = tfidf.transform([input_resume])
   predicted_category = clf.predict(transformed_input)
   print("Predicted Category:", predicted_category)
   ```


## Results

- Achieved **X% accuracy** on the test set.
- Successfully classified resumes into **23 predefined job categories**.
- Visualized category distribution and parsing outputs.


## Folder Structure
```
resume-parser-classifier/
├── data/
│   └── UpdatedResumeDataSet.csv
├── models/
│   ├── tfidf.pkl
│   └── clf.pkl
├── notebooks/
│   └── EDA.ipynb
├── scripts/
│   └── main.py
└── README.md
```


## License

This project is licensed under the [MIT License](LICENSE).


## Acknowledgements
- [spaCy](https://spacy.io/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)



## Contributions
Feel free to submit issues or feature requests. 


## Contact
For queries, reach out via LinkedIn or email at alfinafhobi756@gmail.com.

