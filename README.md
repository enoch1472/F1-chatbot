# Formula 1 Chatbot System

Author: Enoch Obikoya 


## Overview

In this project, I developed a Formula 1 chatbot system that combines knowledge representation, fuzzy logic inference, statistical information retrieval, computer vision models, and natural language processing transformers.

My system allows users to query Formula 1 information, verify facts through logical reasoning, classify F1 car liveries, predict championship outcomes based on driver parameters, optimize race travel calendars, and generate F1 commentary based on user tone.


## Required Libraries and Dependencies

Below is the complete list of Python packages required to run all components of my project:

- tensorflow
- nltk
- simpful
- scikit-learn
- pandas
- numpy
- aiml
- beautifulsoup4
- requests
- azure-cognitiveservices-vision-computervision
- msrest
- pillow
- opencv-python
- matplotlib

### Installation Command

You can install all required dependencies by running:

pip install tensorflow nltk simpful scikit-learn pandas numpy aiml beautifulsoup4 requests azure-cognitiveservices-vision-computervision msrest pillow opencv-python matplotlib

Note: The standard Python libraries used (os, sys, warnings, csv, re, tkinter, math, io, glob) are included in the standard Python installation.

---

## System Modules

### 1. AIML Conversational Kernel
- Main file: mybot-basic.aiml
- I used aiml.Kernel to implement rule-based greeting patterns, conversational handlers, and pattern-based command routing.

### 2. First-Order Logic and Knowledge Base
- Files: kb.csv, mybot-f1.py
- I built a knowledge base initialized from kb.csv using First-Order Logic expressions.
- I integrated NLTK ResolutionProver to prove facts by contradiction and implemented forward chaining to infer new logical rules such as identifying drivers racing for Italian teams.
- I included safety checks to prevent knowledge base corruption when users input new facts that contradict existing rules.

### 3. TF-IDF and Cosine Similarity Retrieval
- Files: qa.csv, mybot-f1.py
- For open-ended question answering, I vectorized my question-answer dataset qa.csv using TfidfVectorizer from scikit-learn.
- User input is matched using Cosine Similarity against preprocessed queries to retrieve accurate answers.

### 4. Mamdani Fuzzy Logic Systems
- Dependency: simpful
- I built a Mamdani fuzzy inference system to estimate driver championship win likelihoods based on Skill (0 to 10) and Experience (0 to 20 years).
- I built a second fuzzy system to translate continuous Softmax output probabilities from my CNN into qualitative confidence labels (Low, Medium, High).

### 5. Local Convolutional Neural Network (CNN)
- Files: Enoch_F1_trained.h5, train_cnn.py
- I designed and trained a Keras CNN model to classify F1 car liveries across 10 constructor teams: AlphaTauri, Aston Martin, Ferrari, Lotus, McLaren, Mercedes, Racing Point, Red Bull, Renault, Williams.
- When triggered, a Tkinter file dialogue opens to allow image selection.

### 6. Azure Cloud Vision Integration
- File: my_azure_cognitve.py
- I integrated Microsoft Azure Computer Vision API to perform cloud-based image analysis, sponsor/brand detection, and automatic image captioning.

### 7. Race Calendar Optimizer
- File: mybot-f1.py
- I implemented a search algorithm to order race locations efficiently and minimize total calendar travel distance.

### 8. Custom Transformer LM and Sentiment Analysis
- File: transformer_nlp.py
- I built a custom Tiny Transformer Language Model from scratch in TensorFlow/Keras incorporating my own PositionalEncoding and MultiHeadSelfAttention layers.
- The model is trained on Formula 1 text extracted from Wikipedia.
- I also implemented a sentiment analyzer to adjust response tone based on positive or negative keywords in user messages.

---

## Running the Application

### Running the Interactive Chatbot

To launch my chatbot, run:

python mybot-f1.py

### Re-training the Local CNN Model

If you wish to train or update my F1 car classification CNN model:

python train_cnn.py


## User Commands and Usage Examples

When my chatbot is running, type commands into the shell prompt:

### General Questions
- Hello
- What is F1
- Who is the current F1 world champion
- What team does Lando Norris drive for

### First-Order Logic Operations
- I know that Hamilton drives for Ferrari
- Check that Hamilton drives for Ferrari

### Image Classification (Local CNN)
- Classify image
- What car is this
- Classify this F1 car

### Cloud Image Classification (Azure)
- Cloud analyze image
- Azure analysis

### Fuzzy Championship Likelihood
- How likely is Verstappen to win the championship
- How likely skill 8 experience 12

### Race Calendar Optimization
- Optimize race calendar
- What is the best order for races

### Sentiment Analysis and Transformer Commentary
- How do I feel
- Tell me about Monza

### Exit Session
- Bye

---

## Project Structure

- mybot-f1.py: Main script and interactive interface driver
- mybot-basic.aiml: AIML pattern rules and routing logic
- transformer_nlp.py: Custom transformer model and sentiment analyzer
- my_azure_cognitve.py: Azure Cognitive Services Vision script
- train_cnn.py: Training script for F1 car CNN model
- Enoch_F1_trained.h5: Trained Keras CNN weights file
- kb.csv: Knowledge base facts file
- qa.csv: Question and answer dataset file
- README.md: Project documentation
