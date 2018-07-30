from django import template
import re

register = template.Library()

@register.filter
def multiply(nb):
    return round(nb*100, 2)

@register.filter
def replaceModel(modelgen):
    li = modelgen.split(' ')
    if (len(li) == 3):
        return  li[0].capitalize() + ' ' + li[1]
    elif(len(li) == 2):
        return li[0].capitalize()

