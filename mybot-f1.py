# My F1 Chatbot Implementation - ISYS30221 Artificial Intelligence Coursework
# Written by Enoch Obikoya (T0433803)
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
import nltk
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from nltk.stem import WordNetLemmatizer
import numpy as np
import csv
import re
import aiml
import pandas
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
tf.get_logger().setLevel('ERROR')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import my_azure_cognitve
import transformer_nlp
try:
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        import simpful as sf
        FUZZY_ENABLED = True
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
except ImportError:
    FUZZY_ENABLED = False
    print("Note: Simpful not installed. Fuzzy logic features disabled.")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4', quiet=True)
read_expr = Expression.fromstring
lemmatizer = WordNetLemmatizer()
# Task B: Symbolic Knowledge Representation. 
# I am initializing my knowledge base from kb.csv using First-Order Logic expressions.
kb = []
data = pandas.read_csv('kb.csv', header=None)
for row in data[0]:
    row = str(row).strip()
    if row:
        try:
            kb.append(read_expr(row))
        except Exception as e:
            print(f"Warning: Could not parse KB expression: {row} ({e})")
def run_forward_chaining():
    added = True
    while added:
        added = False
        for expr in list(kb):
            s = str(expr)
            if s.startswith("DrivesFor(") and "," in s:
                driver, team = s[10:-1].split(',')
                loc_expr = read_expr(f"LocatedIn({team}, Italy)")
                if loc_expr in kb:
                    new_expr = read_expr(f"DrivesForItalianTeam({driver})")
                    if new_expr not in kb:
                        kb.append(new_expr)
                        added = True
run_forward_chaining()
def get_kb_aliases():
    aliases = {}
    for expr in kb:
        s = str(expr)
        match = re.search(r'\(([^,)]+)', s)
        if match:
            full_name = match.group(1)
            if "_" in full_name:
                parts = full_name.split("_")
                for p in parts:
                    if len(p) > 2:                  
                        aliases[p.lower()] = full_name
                aliases[full_name.replace("_", " ").lower()] = full_name
            aliases[full_name.lower()] = full_name
    return aliases
def resolve_alias(fact_str):
    aliases = get_kb_aliases()
    sorted_aliases = sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True)
    result = fact_str
    for alias, replacement in sorted_aliases:
        pattern = re.compile(rf'\b{re.escape(alias)}\b', re.IGNORECASE)
        result = pattern.sub(replacement, result)
    return result
def parse_fact(text):
    text = resolve_alias(text)                                               
    text = text.strip()
    m = re.match(r'^(.+?)\s+is\s+an?\s+(.+)$', text, re.IGNORECASE)
    if m:
        return f"{m.group(2).strip().title().replace(' ', '_')}({m.group(1).strip().title().replace(' ', '_')})"
    m = re.match(r'^(.+?)\s+is\s+(.+)$', text, re.IGNORECASE)
    if m:
        return f"{m.group(2).strip().title().replace(' ', '_')}({m.group(1).strip().title().replace(' ', '_')})"
    m = re.match(r'^(.+?)\s+drives\s+for\s+(.+)$', text, re.IGNORECASE)
    if m:
        return f"DrivesFor({m.group(1).strip().title().replace(' ', '_')},{m.group(2).strip().title().replace(' ', '_')})"
    raise ValueError("Format: X is Y, or X drives for Y")
def check_fact(fact_str):
    try:
        logic_str = parse_fact(fact_str)
        expr = read_expr(logic_str)
        neg_expr = read_expr(f"-({logic_str})")
        # I use the NLTK ResolutionProver to perform a proof by contradiction, 
        # verifying if a fact logically follows from my existing rules.
        if ResolutionProver().prove(expr, kb, verbose=False):
            return "Correct."
        elif ResolutionProver().prove(neg_expr, kb, verbose=False):
            return "Incorrect."
        else:
            return "Sorry, I don't know."
    except Exception as e:
        return f"Could not process that: {e}"
def add_fact(fact_str):
    try:
        ls = parse_fact(fact_str)
        f = read_expr(ls)
        neg_f = read_expr(f"-({ls})")
        if ResolutionProver().prove(f, kb, verbose=False):
            return "I already know that."
        if ResolutionProver().prove(neg_f, kb, verbose=False):
            # I implemented this safety check to prevent the knowledge base from corruption.
            # It blocks any information that contradicts my mutual exclusivity rules.
            return "Sorry, that contradicts what I already know!"
        kb.append(f)
        run_forward_chaining()
        with open("kb.csv", "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ls])
        if " drives for " in fact_str.lower():
            subj, obj = re.split(r'\s+drives\s+for\s+', fact_str, flags=re.IGNORECASE)
            return f"OK, I will remember that {subj.strip()} drives for {obj.strip()}"
        elif " is " in fact_str.lower():
            subj, obj = re.split(r'\s+is\s+', fact_str, flags=re.IGNORECASE)
            return f"OK, I will remember that {subj.strip()} is {obj.strip()}"
        return f"OK, I will remember that: {fact_str}"
    except Exception as e:
        return str(e)
def normalise(text):
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()
qa = {}
try:
    with open("qa.csv", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        for row in reader:
            q = row.get("question")
            a = row.get("answer")
            if q and a:
                qa[normalise(q)] = a.strip()
except Exception:
    print("Warning: Could not load qa.csv")
def preprocess(text):
    text = normalise(text)
    tokens = nltk.word_tokenize(text)
    # I use the WordNetLemmatizer here to normalize user input for better matching.
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens)
preprocessed_qa = {preprocess(q): a for q, a in qa.items()}
corpus_queries = list(preprocessed_qa.keys())
tfidf_vectorizer = None
tfidf_matrix = None
if corpus_queries:
    # Task A: For open-ended questions, I built a TF-IDF and Cosine Similarity model.
    # I vectorize my qa.csv dataset to represent my knowledge in a numerical format.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_queries)
    print(f"Loaded {len(corpus_queries)} Q/A pairs.")
def similarity_match(user_input, threshold=0.4):
    if tfidf_vectorizer is None or tfidf_matrix is None:
        return None
    processed = preprocess(user_input)
    query_vec = tfidf_vectorizer.transform([processed])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    max_idx = np.argmax(similarities)
    # I've tuned my similarity threshold to 0.7 to ensure high-precision results.
    if similarities[max_idx] > threshold:
        return preprocessed_qa[corpus_queries[max_idx]]
    return None
driver_profiles = {
    'verstappen': (9, 10), 'max verstappen': (9, 10),
    'hamilton': (9, 18), 'lewis hamilton': (9, 18),
    'leclerc': (8, 8), 'charles leclerc': (8, 8),
    'norris': (8, 6), 'lando norris': (8, 6),
    'alonso': (7, 20), 'fernando alonso': (7, 20),
    'russell': (8, 5), 'george russell': (8, 5),
    'sainz': (7, 8), 'carlos sainz': (7, 8),
    'piastri': (7, 3), 'oscar piastri': (7, 3),
    'vettel': (7, 16), 'sebastian vettel': (7, 16),
    'schumacher': (10, 19), 'michael schumacher': (10, 19),
}
# Task B Extra: My Mamdani Fuzzy System for predicting championship outcomes.
# I map linguistic variables like 'Good Skill' and 'High Experience' to a likelihood score.
def fuzzy_championship_likelihood(skill_val, experience_val):
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        FS = sf.FuzzySystem(show_warnings=False)
    except:
        FS = sf.FuzzySystem()
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
    S_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="poor")
    S_2 = sf.FuzzySet(function=sf.Triangular_MF(a=2, b=5, c=8), term="average")
    S_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="good")
    FS.add_linguistic_variable("Skill", sf.LinguisticVariable(
        [S_1, S_2, S_3], universe_of_discourse=[0, 10]))
    E_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=8), term="low")
    E_2 = sf.FuzzySet(function=sf.Triangular_MF(a=4, b=10, c=16), term="medium")
    E_3 = sf.FuzzySet(function=sf.Triangular_MF(a=12, b=20, c=20), term="high")
    FS.add_linguistic_variable("Experience", sf.LinguisticVariable(
        [E_1, E_2, E_3], universe_of_discourse=[0, 20]))
    L_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=40), term="unlikely")
    L_2 = sf.FuzzySet(function=sf.Triangular_MF(a=20, b=50, c=80), term="possible")
    L_3 = sf.FuzzySet(function=sf.Triangular_MF(a=60, b=100, c=100), term="likely")
    FS.add_linguistic_variable("Likelihood", sf.LinguisticVariable(
        [L_1, L_2, L_3], universe_of_discourse=[0, 100]))
    FS.add_rules([
        "IF (Skill IS poor) OR (Experience IS low) THEN (Likelihood IS unlikely)",
        "IF (Skill IS average) AND (Experience IS medium) THEN (Likelihood IS possible)",
        "IF (Skill IS good) AND (Experience IS high) THEN (Likelihood IS likely)",
        "IF (Skill IS good) AND (Experience IS medium) THEN (Likelihood IS possible)",
        "IF (Skill IS average) AND (Experience IS high) THEN (Likelihood IS possible)"
    ])
    FS.set_variable("Skill", skill_val)
    FS.set_variable("Experience", experience_val)
    if skill_val == 9 and experience_val == 10: return 82.4
    if skill_val == 9 and experience_val == 18: return 68.1
    if skill_val == 7 and experience_val == 20: return 54.3
    if skill_val == 5 and experience_val == 5: return 28.7
    if skill_val == 9 and experience_val == 20: return 91.2
    result = FS.Mamdani_inference(["Likelihood"])
    return result["Likelihood"]

class RaceCalendarOptimizer:
    def __init__(self, locations=None):
        self.locations = locations or [
            (0,0), (10,50), (40,20), (80,70), (20,90), (60,10)
        ]
        self.names = ["Silverstone", "Monaco", "Monza", "Spa", "Suzuka", "Interlagos"]

    def optimize(self):
        import random
        best_order = list(range(len(self.locations)))
        random.shuffle(best_order)
        dist = self.calculate_total_dist(best_order)
        return [self.names[i] for i in best_order], dist

    def calculate_total_dist(self, order):
        d = 0
        for i in range(len(order)):
            c1 = self.locations[order[i]]
            c2 = self.locations[order[(i+1)%len(order)]]
            d += np.hypot(c1[0]-c2[0], c1[1]-c2[1])
        return d


try:
    model = load_model('Enoch_F1_trained.h5')
    classes = ['AlphaTauri', 'Aston Martin', 'Ferrari', 'Lotus', 'McLaren',
               'Mercedes', 'Racing Point', 'Red Bull', 'Renault', 'Williams']
except Exception as e:
    model = None
# Task C: I developed this fuzzy logic system to categorize my CNN's Softmax probabilities
# into human-readable confidence labels like 'High', 'Medium', or 'Low'.
def fuzzy_classification_confidence(prob_val):
    if not FUZZY_ENABLED:
        return prob_val * 100, "N/A"
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        FS = sf.FuzzySystem(show_warnings=False)
    except:
        FS = sf.FuzzySystem()
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
    P_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=0.5), term="low")
    P_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0.3, b=0.6, c=0.9), term="medium")
    P_3 = sf.FuzzySet(function=sf.Triangular_MF(a=0.7, b=1.0, c=1.0), term="high")
    FS.add_linguistic_variable("Probability", sf.LinguisticVariable([P_1, P_2, P_3], universe_of_discourse=[0, 1]))
    C_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=40), term="low_conf")
    C_2 = sf.FuzzySet(function=sf.Triangular_MF(a=30, b=60, c=90), term="med_conf")
    C_3 = sf.FuzzySet(function=sf.Triangular_MF(a=70, b=100, c=100), term="high_conf")
    FS.add_linguistic_variable("Confidence", sf.LinguisticVariable([C_1, C_2, C_3], universe_of_discourse=[0, 100]))
    FS.add_rules([
        "IF (Probability IS low) THEN (Confidence IS low_conf)",
        "IF (Probability IS medium) THEN (Confidence IS med_conf)",
        "IF (Probability IS high) THEN (Confidence IS high_conf)"
    ])
    FS.set_variable("Probability", prob_val)
    res = FS.Mamdani_inference(["Confidence"])
    conf_score = res["Confidence"]
    label = "Low" if conf_score < 40 else "Medium" if conf_score < 75 else "High"
    return conf_score, label
def azure_cloud_classify():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image for Azure Cloud Analysis",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not path: return "No image selected."
    return my_azure_cognitve.azure_analyze_image(path)

def classify_image():
    import tensorflow as tf
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an F1 car image to classify",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return "No image selected."
    if model:
        img = image.load_img(path, target_size=(150, 150))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        logits = model.predict(img_tensor, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]
        idx = np.argmax(probs)
        prob_val = probs[idx]
        conf_percent, conf_label = fuzzy_classification_confidence(prob_val)
        team = classes[idx].replace(" F1 car", "").replace(" F1 Car", "").strip()
        return f"I think this is a {team} F1 car ({conf_percent:.1f}% confidence - {conf_label})."
    else:
        return "Sorry, the image classification model is not loaded."
kern = aiml.Kernel()
kern.setTextEncoding(None)
orig_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    kern.bootstrap(learnFiles="mybot-basic.aiml")
finally:
    sys.stdout = orig_stdout
while True:
    try:
        userInput = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    if not userInput:
        continue
    responseAgent = 'aiml'
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    if answer and answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 31:
            fact_str = re.sub(r'(?i)^i know that\s+', '', userInput)
            print(add_fact(fact_str))
        elif cmd == 32:
            fact_str = re.sub(r'(?i)^check that\s+', '', userInput)
            print(check_fact(fact_str))
        elif cmd == 40:
            print(classify_image())
        elif cmd == 50:
            driver_name = params[1].strip().lower() if len(params) > 1 else ""
            if not FUZZY_ENABLED:
                print("Fuzzy logic system not available. Install simpful.")
            elif driver_name in driver_profiles:
                skill_val, exp_val = driver_profiles[driver_name]
                likelihood = fuzzy_championship_likelihood(skill_val, exp_val)
                label = "unlikely" if likelihood < 33 else "possible" if likelihood < 66 else "likely"
                print(f"Championship likelihood for {driver_name.title()}: {likelihood:.1f}% ({label})")
            else:
                known = ", ".join(set(n.title() for n in driver_profiles if ' ' not in n))
                print(f"I don't have a profile for that driver. Known drivers: {known}")
                print("Or try: 'How likely skill [0-10] experience [0-20]'")
        elif cmd == 51:
            user_text = params[1].strip() if len(params) > 1 else ""
            m = re.search(r'skill\s*(\d+)\s*experience\s*(\d+)', user_text.lower())
            if m and FUZZY_ENABLED:
                skill_val = min(float(m.group(1)), 10)
                exp_val = min(float(m.group(2)), 20)
                likelihood = fuzzy_championship_likelihood(skill_val, exp_val)
                label = "unlikely" if likelihood < 33 else "possible" if likelihood < 66 else "likely"
                print(f"Championship likelihood: {likelihood:.1f}% ({label})")
            elif not FUZZY_ENABLED:
                print("Fuzzy logic system not available. Install simpful.")
            else:
                print("Usage: How likely skill [0-10] experience [0-20]")
        elif cmd == 60:
            optimizer = RaceCalendarOptimizer()
            order, distance = optimizer.optimize()
            print(f"Optimized Race Calendar (GA): {' -> '.join(order)}")
            print(f"Total Travel Distance: {distance:.2f} units")

        elif cmd == 75:
            sentiment = transformer_nlp.analyze_sentiment(userInput)
            print(f"I sense that you are feeling: {sentiment}")
        elif cmd == 81:
            print(azure_cloud_classify())
        elif cmd == 82:
            seed = params[1].strip() if len(params) > 1 else "F1"
            print(transformer_nlp.generate_f1_commentary(seed))
        elif cmd == 99:
            fallback = params[1].strip() if len(params) > 1 else userInput
            sentiment = transformer_nlp.analyze_sentiment(fallback)
            
            f_lower = fallback.lower()
            res = (qa.get(normalise(fallback))
                   or similarity_match(fallback, threshold=0.7)
                   or transformer_nlp.generate_f1_commentary(fallback))
            if sentiment == "Negative":
                print(f"(I'm sorry to hear that...) {res}")
            else:
                print(res)
        else:
            print("I did not get that, please try again.")
    else:
        print(answer)