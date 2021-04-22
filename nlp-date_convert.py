# This program create API for conversion of natural language date and time into numeric form.
# ---------------------------Possible errors while running---------------------------
# The error like from werkzeug import cached_property,  ImportError: cannot import name 'cached_property'. if this arises the best solution is to 
# downgrade the from werkzeug to 0.16.1 using the command " pip3 install --upgrade Werkzeug==0.16.1 "
# Or you can edit each file with error and change "from werkzeug import cached_property"  to "from werkzeug import.utlis cached_property" 
# Type the following code in terminal to run the api
# FLASK_APP=nlp-date_convert.py flask run --host=localhost

import spacy
from spacy import displacy

from timefhuman import timefhuman

#python -m spacy download en_core_web_sm
nlp = spacy.load('ner-date')

from flask import Flask, request
from flask_restplus import Api, Resource, fields


flask_app = Flask(__name__)
app = Api(app = flask_app,
                  version = "1.0.0 Beta",
                  title = "Implementation of Natural Language Procesing",
                  description = "Conversion of date in text into number using NLP")

def update_sentence(te): 
    newString = te
    doc = nlp(te)
    for e in reversed(doc.ents): #reversed to not modify the offsets of other entities when substituting
      if e.label_ == 'DATE':
        start = e.start_char
        end = start + len(e.text)
        txt = e.text
        y = timefhuman(txt.lower())
        y = str(y.strftime("%d/%m/%Y")).strip('[]')
        newString = newString[:start]+y+newString[end:]
    return newString

name_space = app.namespace('sentence', description='Convert date in text into number using NLP')

@name_space.route("/<text>")
class MainClass(Resource):

      @app.doc(responses={ 200: 'Success', 400: 'Invalid Request', 500: 'Server Error' }, params={ 'text': 'Input is REQUIRED' })

      def get(self, text):
          try:
              return {
                        "status": "Conversion achieved",
                        "name" : update_sentence(text)
                        }
          except KeyError as e:
                  name_space.abort(500, e.__doc__, status = "Not able to proceed. Something is wrong with server", statusCode = "500")
          except Exception as e:
                  name_space.abort(400, e.__doc__, status = "Not able to proceed. Something is wrong with your Script/Request", statusCode = "400")
