# F1 Chatbot - ISYS30221 Artificial Intelligence Coursework - Enoch Obikoya (T0433803)
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
import time
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter
import aiml
import pandas
import tkinter as tk
from tkinter import filedialog
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    tf.get_logger().setLevel('ERROR')
except ImportError:
    tf = None
    load_model = None
    image = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import my_azure_cognitve
except ImportError:
    my_azure_cognitve = None
try:
    import transformer_nlp
except ImportError:
    transformer_nlp = None

def get_sentiment(text):
    if transformer_nlp is not None and hasattr(transformer_nlp, 'analyze_sentiment'):
        return transformer_nlp.analyze_sentiment(text)
    text = text.lower()
    score = 0
    words = text.split()
    for w in words:
        if w in {'great', 'win', 'champion', 'fast', 'amazing', 'happy', 'love', 'perfect', 'best'}: score += 1
        if w in {'bad', 'crash', 'fail', 'slow', 'sad', 'hate', 'lose', 'worst', 'terrible'}: score -= 1
    if score > 0: return "Positive"
    elif score < 0: return "Negative"
    else: return "Neutral"

def get_commentary(seed):
    if transformer_nlp is not None and hasattr(transformer_nlp, 'generate_f1_commentary'):
        return transformer_nlp.generate_f1_commentary(seed)
    return "Formula 1 is the pinnacle of motorsport, featuring single-seater cars racing worldwide since 1950."
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
kb = []
data = pandas.read_csv('kb.csv', header=None)
for row in data[0]:
    row = str(row).strip()
    if row:
        try:
            kb.append(read_expr(row))
        except Exception as e:
            print(f"Warning: Could not parse KB expression: {row} ({e})")
if os.path.exists('user_facts.csv'):
    try:
        udata = pandas.read_csv('user_facts.csv', header=None)
        for row in udata[0]:
            row = str(row).strip()
            if row:
                try:
                    kb.append(read_expr(row))
                except Exception:
                    pass
    except Exception:
        pass
def run_forward_chaining():
    added = True
    while added:
        added = False
        for expr in list(kb):
            s = str(expr)
            if s.startswith("DrivesFor(") and "," in s:
                driver, team = s[10:-1].split(',')
                loc_expr = read_expr(f"LocatedIn({team.strip()}, Italy)")
                if loc_expr in kb:
                    new_expr = read_expr(f"DrivesForItalianTeam({driver.strip()})")
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
            full_name = match.group(1).strip()
            if full_name.endswith("_GP") or full_name in {'Driver', 'Team', 'Circuit', 'Race'}:
                continue
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
    m_it = re.match(r'^(.+?)\s+drives\s+for\s+an?\s+italian\s+team$', text, re.IGNORECASE)
    if m_it:
        driver = resolve_alias(m_it.group(1).strip()).title().replace(' ', '_')
        return f"DrivesForItalianTeam({driver})"
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
def _get_drives_for(driver_canonical):
    """Return the team a driver drives for in the KB, or None."""
    for expr in kb:
        s = str(expr)
        if s.startswith("DrivesFor(") and "," in s:
            inner = s[len("DrivesFor("):-1]
            parts = inner.split(",", 1)
            if len(parts) == 2 and parts[0].strip() == driver_canonical:
                return parts[1].strip()
    return None

def _known_entities():
    """Return all constants that appear as subject of any KB fact."""
    entities = set()
    for expr in kb:
        s = str(expr)
        m = re.match(r'^[A-Za-z_]+\(([^,)]+)', s)
        if m:
            entities.add(m.group(1).strip())
    return entities

def check_fact(fact_str):
    try:
        logic_str = parse_fact(fact_str)
        expr = read_expr(logic_str)
        neg_expr = read_expr(f"-({logic_str})")
        if logic_str.startswith("DrivesFor("):
            inner = logic_str[len("DrivesFor("):-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                driver_c, claimed_team = parts[0].strip(), parts[1].strip()
                actual_team = _get_drives_for(driver_c)
                if actual_team is not None:
                    return "Correct." if actual_team == claimed_team else "Incorrect."
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
        m_subject = re.match(r'^[A-Za-z_]+\(([^,)]+)', ls)
        if m_subject:
            subject = m_subject.group(1).strip()
            if subject not in _known_entities():
                return (f"I don't recognise '{subject.replace('_', ' ')}'. "
                        "I can only learn facts about entities already in my knowledge base.")
        if ls.startswith("DrivesFor("):
            inner = ls[len("DrivesFor("):-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                driver_c, claimed_team = parts[0].strip(), parts[1].strip()
                actual_team = _get_drives_for(driver_c)
                if actual_team is not None:
                    if actual_team == claimed_team:
                        return "I already know that."
                    else:
                        return (f"Sorry, that contradicts what I already know! "
                                f"({driver_c.replace('_', ' ')} drives for {actual_team})")
        if ResolutionProver().prove(f, kb, verbose=False):
            return "I already know that."
        kb.append(f)
        run_forward_chaining()
        with open("user_facts.csv", "a", newline='') as file:
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
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens)
preprocessed_qa = {preprocess(q): a for q, a in qa.items()}
corpus_queries = list(preprocessed_qa.keys())
tfidf_vectorizer = None
tfidf_matrix = None
if corpus_queries:
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
    if similarities[max_idx] > threshold:
        return preprocessed_qa[corpus_queries[max_idx]]
    return None
driver_profiles = {
    'verstappen': (9, 10), 'max verstappen': (9, 10),
    'perez': (7, 13), 'sergio perez': (7, 13),
    'hamilton': (9, 17), 'lewis hamilton': (9, 17),
    'russell': (8, 5), 'george russell': (8, 5),
    'leclerc': (8, 7), 'charles leclerc': (8, 7),
    'sainz': (7, 9), 'carlos sainz': (7, 9),
    'norris': (8, 6), 'lando norris': (8, 6),
    'piastri': (7, 2), 'oscar piastri': (7, 2),
    'alonso': (7, 20), 'fernando alonso': (7, 20),
    'stroll': (5, 7), 'lance stroll': (5, 7),
    'gasly': (6, 7), 'pierre gasly': (6, 7),
    'ocon': (6, 7), 'esteban ocon': (6, 7),
    'albon': (6, 5), 'alexander albon': (6, 5),
    'sargeant': (4, 2), 'logan sargeant': (4, 2),
    'tsunoda': (6, 4), 'yuki tsunoda': (6, 4),
    'ricciardo': (7, 13), 'daniel ricciardo': (7, 13),
    'hulkenberg': (6, 10), 'nico hulkenberg': (6, 10),
    'magnussen': (5, 9), 'kevin magnussen': (5, 9),
    'bottas': (6, 11), 'valtteri bottas': (6, 11),
    'zhou': (5, 3), 'zhou guanyu': (5, 3),
    # Historical legends
    'vettel': (7, 16), 'sebastian vettel': (7, 16),
    'schumacher': (10, 19), 'michael schumacher': (10, 19),
}
def fuzzy_championship_likelihood(skill_val, experience_val):
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        FS = sf.FuzzySystem(show_banner=False)
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
    result = FS.Mamdani_inference(["Likelihood"])
    return result["Likelihood"]

class RaceCalendarOptimizer:
    """
    Genetic Algorithm (GA) optimizer for the Travelling Salesperson Problem (TSP)
    applied to ordering Formula 1 Grand Prix calendar locations to minimise total travel distance.
    """
    def __init__(self, locations=None, names=None, pop_size=50, generations=100, mutation_rate=0.15, elite_size=2):
        self.locations = locations or [
            (0,0), (10,50), (40,20), (80,70), (20,90), (60,10)
        ]
        self.names = names or ["Silverstone", "Monaco", "Monza", "Spa", "Suzuka", "Interlagos"]
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def calculate_total_dist(self, order):
        d = 0.0
        n = len(order)
        for i in range(n):
            c1 = self.locations[order[i]]
            c2 = self.locations[order[(i+1) % n]]
            d += np.hypot(c1[0]-c2[0], c1[1]-c2[1])
        return d

    def _crossover_ox(self, parent1, parent2):
        """Order Crossover (OX) preserving permutation validity for TSP."""
        import random
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[a:b+1] = parent1[a:b+1]
        
        current_idx = (b + 1) % size
        for gene in parent2[b+1:] + parent2[:b+1]:
            if gene not in child:
                child[current_idx] = gene
                current_idx = (current_idx + 1) % size
        return child

    def _mutate(self, order):
        """Inversion (2-opt style) mutation to reverse an edge sub-sequence."""
        import random
        if random.random() < self.mutation_rate:
            a, b = sorted(random.sample(range(len(order)), 2))
            order[a:b+1] = reversed(order[a:b+1])
        return order

    def _tournament_select(self, population, k=3):
        """Tournament selection selecting the tour with the lowest travel distance."""
        import random
        selected = random.sample(population, k)
        return min(selected, key=self.calculate_total_dist)

    def optimize(self):
        """
        Executes the Genetic Algorithm:
        - Population initialization with random permutation tours
        - Generational evolution with Elitism, OX crossover, and inversion mutation
        - Returns the best circuit order and minimum total distance found.
        """
        import random
        n = len(self.locations)
        if n <= 1:
            return self.names, 0.0

        base = list(range(n))
        population = [random.sample(base, n) for _ in range(self.pop_size)]

        best_individual = min(population, key=self.calculate_total_dist)
        best_dist = self.calculate_total_dist(best_individual)

        for _ in range(self.generations):
            population.sort(key=self.calculate_total_dist)
            current_best = population[0]
            current_best_dist = self.calculate_total_dist(current_best)
            if current_best_dist < best_dist:
                best_dist = current_best_dist
                best_individual = current_best[:]

            next_pop = [population[i][:] for i in range(self.elite_size)]

            while len(next_pop) < self.pop_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)
                child = self._crossover_ox(p1, p2)
                child = self._mutate(child)
                next_pop.append(child)

            population = next_pop

        population.sort(key=self.calculate_total_dist)
        if self.calculate_total_dist(population[0]) < best_dist:
            best_dist = self.calculate_total_dist(population[0])
            best_individual = population[0]

        return [self.names[i] for i in best_individual], best_dist


try:
    model = load_model('Enoch_F1_trained.h5')
    classes = ['AlphaTauri', 'Aston Martin', 'Ferrari', 'Lotus', 'McLaren',
               'Mercedes', 'Racing Point', 'Red Bull', 'Renault', 'Williams']
except Exception as e:
    model = None
def fuzzy_classification_confidence(prob_val):
    if not FUZZY_ENABLED:
        return prob_val * 100, "N/A"
    import sys
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    try:
        FS = sf.FuzzySystem(show_banner=False)
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
    if my_azure_cognitve is not None and hasattr(my_azure_cognitve, 'azure_analyze_image'):
        return my_azure_cognitve.azure_analyze_image(path)
    return "Azure Vision module not available. Install azure-cognitiveservices-vision-computervision."

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
            sentiment = get_sentiment(userInput)
            print(f"I sense that you are feeling: {sentiment}")
        elif cmd == 76:
            test_text = params[1].strip() if len(params) > 1 else userInput
            sentiment = get_sentiment(test_text)
            positive_words = {'great', 'win', 'champion', 'fast', 'amazing', 'happy', 'love', 'perfect', 'best'}
            negative_words = {'bad', 'crash', 'fail', 'slow', 'sad', 'hate', 'lose', 'worst', 'terrible'}
            found = [w for w in test_text.lower().split() if w in positive_words | negative_words]
            detail = f" (detected keywords: {', '.join(found)})" if found else ""
            print(f"Sentiment of \"{test_text}\": {sentiment}{detail}")
        elif cmd == 81:
            print(azure_cloud_classify())
        elif cmd == 82:
            seed = params[1].strip() if len(params) > 1 else "F1"
            print(get_commentary(seed))
        elif cmd == 99:
            fallback = params[1].strip() if len(params) > 1 else userInput
            sentiment = get_sentiment(fallback)
            
            f_lower = fallback.lower()
            res = (qa.get(normalise(fallback))
                   or similarity_match(fallback, threshold=0.7)
                   or get_commentary(fallback))
            if sentiment == "Negative":
                print(f"(I'm sorry to hear that...) {res}")
            else:
                print(res)
        else:
            print("I did not get that, please try again.")
    else:
        print(answer)