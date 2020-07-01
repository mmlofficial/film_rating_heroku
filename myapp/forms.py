from django import forms

class NameForm(forms.Form):
    your_text = forms.CharField(label = '', max_length=1000)