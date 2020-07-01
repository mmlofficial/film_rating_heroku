from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from .forms import NameForm

from string import punctuation
from nltk.corpus import stopwords
import pickle5
import os

english_stopwords = stopwords.words("english")
punctuation += '1234567890'
tt = str.maketrans(dict.fromkeys(punctuation))

with open('TFIDF_Vectorizer.pkl', 'rb') as f:
    tfv = pickle5.load(f)

with open('Classifier.pkl', 'rb') as f:
    clf = pickle5.load(f)

def clean_string(text):
    tokens = text.lower().split(' ')
    tok = []
    for token in tokens:
        t = token.translate(tt)
        if t not in english_stopwords and t != ' ' and t != '':
            tok.append(t)
    return ' '.join(tok)

def get_rating_status(pred):
    rating = pred[0]
    if rating < 4:
        rating += 1
        status = 'Отрицательный'
    else:
        rating += 3
        status = 'Положительный'
    return str(rating), status

def index(request):
    # if this is a POST request we need to process the form data
    text = ''
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            text = form.cleaned_data.get('your_text')
            clean_text = clean_string(text)
            text_tfv = tfv.transform([clean_text])
            pred = clf.predict(text_tfv)
            rating, status = get_rating_status(pred)
            # redirect to a new URL:
            return render(request, 'index.html', {'form': form, 'your_text': text, 'rating': rating, 'status': status})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()


    return render(request, 'index.html', {'form': form, 'your_text': text})