from django.shortcuts import render
from django.http import HttpResponse
from .forms import wordForm
from django.shortcuts import redirect
#######################################
import keras
import numpy as np
import random
import sys
import json
import os
import tensorflow as tf
from keras.models import load_model
from numpy.random import seed
seed(1)
path= module_dir = os.path.dirname(__file__)
global fullText, model, graph
fullText = open(path +'\\static\\predictions\\model\\120TaylorSongsLyrics.txt',encoding='utf8').read().lower()
chars = sorted(list(set(fullText)))
char_indices = dict((char, chars.index(char)) for char in chars)

model = load_model(path+'\\static\\predictions\\model\\my_model.h5')
graph = tf.get_default_graph()

maxlen = 20
text2=''
#######################################
# Create your views here.
def index(request):
    #return render(request,'predictions/index.html')
    return redirect('/predict')
    #return HttpResponse('Working :)')

def predict(request):
    text = ''
    text = generateText(text)
    return render(request, 'predictions/index.html',{'text': text})
    '''
    if(request.POST):
        form =  wordForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['word']
            text = generateText(text)

    else:
        form = wordForm()
        text = ''


    args = {'form': form, 'text': text}
    return render(request, 'predictions/index.html',args)
    '''



def generateText(txt):
    txt = txt.lower()
    def sample(preds, temperature=0.5):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    with graph.as_default():
        #generated_text =  txt[0: 0 + maxlen]
        start_index = random.randint(0, len(fullText) - maxlen - 1)
        generated_text = fullText[start_index: start_index + maxlen]
        text2 = generated_text
        print('--- Generating with seed: "' + generated_text + '"')
        temperature = 0.5
        for i in range(200):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            text2+=next_char
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
            sys.stdout.flush()
    c = '"'
    text2 = c+text2+c
    return text2
    print()
