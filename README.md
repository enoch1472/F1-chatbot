Formula 1 Chatbot

Written by Enoch Obikoya


What this project is

This is a command-line chatbot about Formula 1 that I built for my AI coursework module. You talk to it by typing, and it responds with facts, predictions, image analysis and generated commentary. Each feature uses a different AI technique, and they all run together inside one chat loop.


What it can do

When you type something, the bot first tries to match it against a pattern in the AIML file. That handles greetings, goodbyes and the trigger words for each feature. If nothing matches exactly, it falls back to a similarity search or generates a response using the transformer.

The knowledge base stores facts about F1 drivers, teams and circuits as first-order logic in a file called kb.csv. When you ask it to check a fact, it uses NLTK's resolution prover to work out whether that fact can be proven or disproven. If it cannot tell either way, it says so. You can also teach it new facts during the conversation. Before it accepts one, it checks the subject is already in the knowledge base and that the new fact does not contradict anything it already knows. Facts you add are saved to user_facts.csv so they persist.

The fuzzy logic side takes a skill level from zero to ten and an experience value from zero to twenty years, then runs them through a Mamdani fuzzy system built with simpful to give a championship likelihood percentage. A second fuzzy system converts the CNN's raw probability into a Low, Medium or High confidence label.

The genetic algorithm treats the race calendar as a travelling salesman problem across six circuits. It uses tournament selection, order crossover, inversion mutation and elitism to find the shortest route. The total distance it finds is consistently 285.79 units, though the circuit order can vary between runs.

The image classifier is a convolutional neural network trained to recognise ten F1 team liveries. Those are AlphaTauri, Aston Martin, Ferrari, Lotus, McLaren, Mercedes, Racing Point, Red Bull, Renault and Williams. A file browser opens so you can pick any photo, and the network tells you which team it thinks the car belongs to.

The transformer language model is something I built from scratch, including the positional encoding and multi-head self-attention layers. Every time the bot starts, it downloads the Formula One Wikipedia page and trains on it. To stop the output from looping on the same word, I added temperature scaling, top-k sampling and a repetition penalty to the generation step.

For general questions not covered by AIML, the bot searches qa.csv using TF-IDF and cosine similarity to find the best match. This gives it a broad base of F1 knowledge to draw from.

The Azure Computer Vision integration sends an image to the cloud and returns a caption, tags, detected objects and any sponsor brands. You need to add your own Azure key and endpoint to my_azure_cognitve.py for this to work.

The sentiment feature reads the tone of what you type. If it sounds negative, the bot adds an empathetic note before its reply. You can also test it directly by typing something like analyze sentiment followed by any phrase.


How to set it up

You need Python 3.10, 3.11 or 3.12. Install the dependencies by running this in your terminal.

pip install tensorflow nltk simpful scikit-learn pandas numpy aiml beautifulsoup4 requests azure-cognitiveservices-vision-computervision msrest pillow opencv-python matplotlib

The NLTK language data downloads automatically on the first run.


How to start the chatbot

Run this from inside the project folder.

python mybot-f1.py

It needs an internet connection at startup because the transformer downloads and trains on the Wikipedia page each time, so the first launch takes a little longer while that completes.


How to train the image classifier

Put your car images into a folder with one subfolder per team, set the data_dir variable at the top of train_cnn.py to that folder, then run it.

python train_cnn.py

It trains for 15 epochs and saves the model as Enoch_F1_trained.h5. It also writes training_curves.png and confusion_matrix.png and prints a full classification report.


How to test each feature

Once the chatbot is running, try these inputs one at a time.

For general conversation, type Hello, then What is DRS, then Bye.

For the knowledge base, type Check that Hamilton drives for Mercedes and you should get Correct. Type Check that Hamilton drives for Ferrari for Incorrect. Type Check that Leclerc drives for an Italian team and Check that Piastri is Australian for Correct both times. Then type I know that Hamilton drives for Ferrari and it should say that contradicts what it already knows. Type I know that Fast is Lando Norris and it should say it does not recognise Fast.

For fuzzy championship prediction, type How likely is Verstappen to win the championship for around 50 percent, How likely is Hamilton to win the championship for around 85 percent, and How likely skill 7 experience 15 for around 63 percent.

For the race calendar, type Optimize race calendar. The circuits appear in an efficient order and the total distance is 285.79 units.

For image classification, type Classify image and pick a car photo from the file browser. For the Azure version, type Cloud analyze image, though this only works once you have added your credentials.

For the transformer, type Tell me about Monza.

For sentiment testing, type Analyze sentiment I love this race and it will return Positive. Try Analyze sentiment that crash was terrible for Negative.
