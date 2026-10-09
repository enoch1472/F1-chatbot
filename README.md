Formula 1 Chatbot

Written by Enoch Obikoya

What this project is

This is a command-line chatbot about Formula 1 that I built for my AI coursework module. You talk to it by typing, and it responds with facts, predictions, image analysis and generated commentary. Each feature uses a different AI technique, and they all run together inside one chat loop.

When you type something, the bot first tries to match it against a pattern in the AIML file. That handles greetings, goodbyes and the trigger words for each feature. If nothing matches exactly, it falls back to a similarity search or generates a response using the transformer.

Setup

You need Python 3.10, 3.11 or 3.12. Install the dependencies by running this in your terminal:

pip install tensorflow nltk simpful scikit-learn pandas numpy aiml beautifulsoup4 requests azure-cognitiveservices-vision-computervision msrest pillow opencv-python matplotlib

The NLTK language data downloads automatically on the first run.

Starting the chatbot

Run this from inside the project folder:

python mybot-f1.py

It needs an internet connection at startup because the transformer downloads and trains on the Wikipedia page each time, so the first launch takes a little longer while that completes.

Features and how to test them

Once the chatbot is running, try the inputs under each feature one at a time.

General conversation (AIML)

AIML handles greetings, goodbyes and the trigger phrases that send each command to the right part of the program.

Try:

Hello
What is F1
Bye
<img width="1917" height="975" alt="AIML" src="https://github.com/user-attachments/assets/cf9e4739-2eac-4138-aef4-7740a7478bc1" />
Question answering

For general questions not covered by AIML, the bot searches qa.csv using TF-IDF and cosine similarity to find the best match. This gives it a broad base of F1 knowledge to draw from.

Try:

What is DRS
Who has the most F1 championships?
What does a yellow flag mean
<img width="1918" height="968" alt="QA" src="https://github.com/user-attachments/assets/5d9f6010-9fd2-494d-9fbe-b4fe6a30cbd2" />
Knowledge base and logic

The knowledge base stores facts about F1 drivers, teams and circuits as first-order logic in a file called kb.csv. When you ask it to check a fact, it uses NLTK's resolution prover to work out whether that fact can be proven or disproven. If it cannot tell either way, it says so.

You can also teach it new facts during the conversation. Before it accepts one, it checks that the subject is already in the knowledge base and that the new fact does not contradict anything it already knows. Facts you add are saved to user_facts.csv so they persist.

Try:

Check that Hamilton drives for Mercedes
Check that Hamilton drives for Ferrari
Check that Leclerc drives for an Italian team
Check that Piastri is Australian
I know that Hamilton drives for Ferrari
I know that Fast is Lando Norris

These give Correct, Incorrect, Correct, Correct, a message that the fact contradicts what the bot already knows, and a message that it does not recognise "Fast".

<!-- Drag your Logic screenshot here -->
Fuzzy logic

The fuzzy logic side takes a skill level from zero to ten and an experience value from zero to twenty years, then runs them through a Mamdani fuzzy system built with simpful to give a championship likelihood percentage. A second fuzzy system converts the CNN's raw probability into a Low, Medium or High confidence label.

Try:

How likely is Verstappen to win the championship
How likely is Hamilton to win the championship
How likely skill 7 experience 15

You should get around 50, 85 and 63 percent.

<img width="1918" height="921" alt="Fuzzy" src="https://github.com/user-attachments/assets/c084b238-5fc1-48a7-ad67-dca75d9f7c70" />
Genetic algorithm

The genetic algorithm treats the race calendar as a travelling salesman problem across six circuits. It uses tournament selection, order crossover, inversion mutation and elitism to find the shortest route. The total distance it finds is consistently 285.79 units, though the circuit order can vary between runs.

Try:

Optimize race calendar
<!-- Drag your Genetic Algorithm screenshot here -->
Image classification (CNN)

The image classifier is a convolutional neural network trained to recognise ten F1 team liveries: AlphaTauri, Aston Martin, Ferrari, Lotus, McLaren, Mercedes, Racing Point, Red Bull, Renault and Williams. A file browser opens so you can pick any photo, and the network tells you which team it thinks the car belongs to, along with a Low, Medium or High confidence label.

Try:

Classify image
<img width="1918" height="968" alt="CNN" src="https://github.com/user-attachments/assets/9bb20229-7d5f-44f6-a76c-020f14032219" />

Results. On the validation set the model reaches about 78% accuracy with a macro F1 of 0.69. It is strongest on Ferrari, Red Bull, Racing Point and Aston Martin. AlphaTauri is the weakest class because there are far fewer training images for it.

Training it yourself. Put your car images into a folder with one subfolder per team, set the data_dir variable at the top of train_cnn.py to that folder, then run:

python train_cnn.py

It trains for 15 epochs and saves the model as Enoch_F1_trained.h5. It also writes training_curves.png and confusion_matrix.png and prints a full classification report.

Transformer

The transformer language model is something I built from scratch, including the positional encoding and multi-head self-attention layers. Every time the bot starts, it downloads the Formula One Wikipedia page and trains on it. To stop the output from looping on the same word, I added temperature scaling and top-k sampling to the generation step.

It is a next-word predictor, so it continues the words it is given rather than answering a question.

Try:

Tell me about Monza
<img width="1918" height="872" alt="Transformer" src="https://github.com/user-attachments/assets/1eb3a81a-912d-46a1-9fe5-78b80470849e" />
Azure Computer Vision

The Azure integration sends an image to the cloud and returns a caption, tags, detected objects and any sponsor brands. You need to add your own Azure key and endpoint to my_azure_cognitve.py for this to work.

Try:

Cloud analyze image
<img width="1918" height="877" alt="Azure Computer Vision" src="https://github.com/user-attachments/assets/4f693ac5-40c1-41a9-8c41-9673fb1c91c0" />
Sentiment analysis

The sentiment feature reads the tone of what you type using keyword matching. If it sounds negative, the bot adds an empathetic note before its reply.

Try:

Do you think I am happy
Do you think I am sad
I hate slow races

The first two report Positive and Negative. The third goes through the fallback and starts with an empathetic note.

<img width="1918" height="872" alt="Sentiment" src="https://github.com/user-attachments/assets/51aa3eac-169e-4728-807d-b81b32a473e8" />
