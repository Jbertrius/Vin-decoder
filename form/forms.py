from django import forms

class VinrequestForm(forms.Form):

    CHOICES = [('v1', 'Version 1'),
               ('v2', 'Version 2')]

    vin = forms.CharField(max_length=100)
    version = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect())