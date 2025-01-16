# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from resume_parser import resumeparse

# Load the dataset
df = pd.read_csv('/UpdatedResumeDataSet.csv')
print(df.head())
print(df.shape)

# Check the distribution of categories
print(df['Category'].value_counts())

# Plot the distribution of categories
plt.figure(figsize=(15,5))
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()

# Plot pie chart for category distribution
counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()

# Clean resume text function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Apply cleaning function to all resumes in the dataset
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
print(df['Resume'][0])

# Encode category labels
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])
print(df.Category.unique())

# Vectorize the resumes
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
requiredText = tfidf.transform(df['Resume'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)

# Train a K-Nearest Neighbors classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Predict and evaluate the model
ypred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, ypred)}")

# Save the trained models to disk
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))

# Install necessary packages (make sure you have access to a suitable environment)
# !pip install resume-parser
# !pip install https://github.com/explosion/spacy/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
# !python -m spacy download en_core_web_sm
# !python -m nltk.downloader stopwords
# !python -m nltk.downloader punkt
# !python -m nltk.downloader averaged_perceptron_tagger
# !python -m nltk.downloader universal_tagset
# !python -m nltk.downloader wordnet
# !python -m nltk.downloader brown
# !python -m nltk.downloader maxent_ne_chunker

# Set Tika server path explicitly (if required)
# os.environ['TIKA_SERVER_JAR'] = '/tmp/tika-server.jar'

# Example resume parsing
data = resumeparse.read_file("/path_to_resume.pdf")  # Update the path
formatted_text = ', '.join(str(value) for value in data.values() if value)
print(formatted_text)

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))

# Clean the input resume
myresume = formatted_text
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# Make the prediction using the trained classifier
prediction_id = clf.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}
category_name = category_mapping.get(prediction_id, "Unknown")

# Output the predicted category
print("Predicted Category:", category_name)
