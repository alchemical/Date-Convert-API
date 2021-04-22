# Date-Convert-API
This API convert natural language date and time into numeric format.
for example if you provide the input as :
yesterday i went to cinema then you will get output as:
02/04/2020 i went to cinema. Considering that today is 03/04/2020

# I recommed to Create virtual environment for python to test this code.
# Information on training and API code.
First of all you need to install the required library as describe as in **requirement.txt**
Also you need to install **Available trained pipelines for English** using the command *python -m spacy download en_core_web_sm*
After finishing installation of required library then you can run
* python3 train-NER.py * which will create the trained model folder called **ner-date** for further use.
Then you can run API
* FLASK_APP=nlp-date_convert.py flask run --host=localhost *Cancel changes
After running the api in terminal you can see:
**---------**
 Serving Flask app "nlp-date_convert.py"
 Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
  Debug mode: off
  Running on http://localhost:5000/ (Press CTRL+C to quit)
  **----------------**
  copy the adderss of local host and put in your browser you can get the api where you can test with various date and time.
  
  **This API use timesofhuman for conversion of date. This model is trained to find the the date in natural language in the sentence to convert using timesofhuman back into numerical form. Therfore the limitation of the modle based on the trained sentence as well as the timesofhuman conversion.**
  
