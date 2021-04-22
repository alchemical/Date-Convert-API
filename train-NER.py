import spacy

nlp = spacy.load('en_core_web_sm')

ner = nlp.get_pipe("ner")

TRAIN_DATA = [       
               ("I will go to a party upcoming Sunday.", { "entities": [(21, 36, "DATE")]}),
               ("I will go to a party Upcoming sunday.", { "entities": [(21, 36, "DATE")]}),
               ("Upcoming Sunday I am planning to watch live game on the TV.", { "entities": [(0, 15, "DATE")]}),
               ("upcoming Sunday I am planning to watch live game on the TV.", { "entities": [(0, 15, "DATE")]}),
               ("I will meet you upcoming Sunday near metro station.", { "entities": [(16, 31, "DATE")]}),
               ("I will meet you Upcoming Sunday near metro station.", { "entities": [(16, 31, "DATE")]}),
               ("I will meet you upcoming sunday near metro station.", { "entities": [(16, 31, "DATE")]}),
               ("We will go to a party upcoming Wednesday.", { "entities": [(22, 40, "DATE")]}),
               ("We will go to a party upcoming wednesday.", { "entities": [(22, 40, "DATE")]}),
               ("Upcoming Wednesday. I am planning to watch live game on the TV.", { "entities": [(0, 18, "DATE")]}),
               ("upcoming wednesday. I am planning to watch live game on the TV.", { "entities": [(0, 18, "DATE")]}),
               ("I will meet you upcoming wednesday near metro station.", { "entities": [(16, 34, "DATE")]}),      
               ("I will meet you upcoming Wednesday near metro station.", { "entities": [(16, 34, "DATE")]}),          
               ("Thursday will have better weather for playing.", { "entities": [(0, 8, "DATE")]}),
               ("thursday will have better weather for playing.", { "entities": [(0, 8, "DATE")]}),
               ("There must be party every thursday.", { "entities": [(26, 34, "DATE")]}),
               ("Last thursday was a good day.", { "entities": [(0, 13, "DATE")]}),
               ("last Thursday was a good day.", { "entities": [(0, 13, "DATE")]}),
               ("Last Thursday was a good day.", { "entities": [(0, 13, "DATE")]}),
               ("It was a good day on tuesday.", { "entities": [(21, 28, "DATE")]}),
               ("It was a good day on Tuesday.", { "entities": [(21, 28, "DATE")]}),
               ("There was never a good day like yesterday.", { "entities": [(32, 41, "DATE")]}),
               ("There was never a good day like Yesterday.", { "entities": [(32, 41, "DATE")]}),
               ("Day after tomorrow i am going to party.", { "entities": [(0, 18, "DATE")]}),
               ("day After tomorrow i am going to party.", { "entities": [(0, 18, "DATE")]}),
               ("Day after Tomorrow i am going to party.", { "entities": [(0, 18, "DATE")]}),
               ("Day After Tomorrow i am going to party.", { "entities": [(0, 18, "DATE")]}),     
               ("Day After Tomorrow i am going to party and call everyone Day after Tomorrow for the party", { "entities": [(0, 18, "DATE"), (57,75,"DATE")]}),
               ("Day After Tomorrow i am going to party and call everyone day After tomorrow for the party", { "entities": [(0, 18, "DATE"), (57,75,"DATE")]}),
               ("Day After Tomorrow i am going to party and call everyone Day After Tomorrow for the party", { "entities": [(0, 18, "DATE"), (57,75,"DATE")]}),
               ("All of us will come day after tomorrow.", { "entities": [(30, 38, "DATE")]}),
               ("This Tuesday we are playing football with experts.", { "entities": [(0, 12, "DATE")]}),
               ("this tuesday we are playing football with experts.", { "entities": [(0, 12, "DATE")]}),
               ("This tuesday we are playing football with experts.", { "entities": [(0, 12, "DATE")]}),
               ("This Tuesday we are playing football with experts.", { "entities": [(0, 12, "DATE")]}),
               ("They are planning to come here this monday for dinner.", { "entities": [(31, 42, "DATE")]}),   
               ("They are planning to come here This Monday for dinner.", { "entities": [(31, 42, "DATE")]}),
               ("They are planning to come here this Monday for dinner.", { "entities": [(31, 42, "DATE")]}),
             ] 


for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions] 



# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training for 30 iterations
  for iteration in range(500):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
        print("Losses", losses)


# Save the  model to directory
output_dir = Path('ner-date')
nlp.to_disk(output_dir)
print("Saved model to", output_dir)




