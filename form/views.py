from django.shortcuts import render
import numpy as np
import pandas as pd
import operator
from django.core.exceptions import ObjectDoesNotExist
from sklearn.externals import joblib
from .forms import VinrequestForm
from .models import Variante, Vin, vds_map
from functools import reduce

import os.path
import logging

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dict =  joblib.load( os.path.join(BASE, 'ml_model/v2/columns.pkl')  )

logging.basicConfig(filename='debug.log',level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Modele version 2
RdF_v2 = joblib.load( os.path.join(BASE, 'ml_model/v2/model/model_clus.pkl')  )
enc_v2 = joblib.load( os.path.join(BASE, 'ml_model/v2/encoder/enc_clus.pkl')  )

def home(request):
    title = 'VIN identifier'
    show_res = False

    if request.method == 'POST':
        form = VinrequestForm(request.POST or None)

        if form.is_valid():
            vin = form.cleaned_data['vin']
            version = form.cleaned_data['version']

            try:
                _vin = Vin.objects.get(vin=vin)

                if version == 'v1':
                    res = _vin.variante_pred_set.filter(version='v1').order_by('-prob')
                    if res.count() == 0:
                        decodev1(vin )
                elif version == 'v2':
                    res = _vin.variante_pred_set.filter(version='v2').order_by('-prob')
                    if res.count() == 0:
                        decodev2(vin )

            except ObjectDoesNotExist:
                if version == 'v1':
                    res = decodev1(vin)
                elif version == 'v2':
                    res = decodev2(vin)
            show_res = True
    else:
        form = VinrequestForm()

    return render(request, 'index.html', locals())

def fn_v2( _str ):
    if ( len(_str) == 17 ):
        return np.array([[_str[:3], _str[3:6], _str[6:8], _str[8], _str[-8], _str[-7], _str[-6:] ]], dtype='object')
    else:
        return []

def fn_v1( _str ):
    if ( len(_str) == 17 ):
        return np.array([[_str[:3], _str[10:11], _str[8:9], _str[3:8], _str[9:10], _str[-6:] ]], dtype='object')
    else:
        return []

def decodev1( vin  ):
    to_save = False
    try:
        _vin = Vin.objects.get(vin=vin)
    except ObjectDoesNotExist:
        _vin = Vin(vin=vin)
        to_save = True

    vin_array = fn_v1(vin)


    RdF = joblib.load(  os.path.join(BASE, 'ml_model/v1/Randomforest.pkl.z')  )
    enc = joblib.load(  os.path.join(BASE, 'ml_model/v1/enc_0.pkl') )

    test = pd.DataFrame(vin_array, columns=enc.cols + ['seq'])
    pred_cluster = RdF.predict_proba(enc.transform(test).drop(columns=['checkD_0', 'checkD_1', 'checkD_2', 'checkD_3', 'checkD_4', 'checkD_5']))

    x = dict(zip( RdF.classes_ + 1, pred_cluster.tolist()[0] ))

    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    tab = [ x for x in sorted_x[:2] ]
    _tab = []
    delete_var(RdF)

    for itm in tab:
        RdF_2e = joblib.load( os.path.join(BASE, 'ml_model/v1/model_' + str(itm[0]) + '.pkl' ) )
        enc_2e = joblib.load( os.path.join(BASE, 'ml_model/v1/enc_' + str(itm[0]) + '.pkl' ) )

        pred_variante = RdF_2e.predict_proba(
            enc_2e.transform(pd.DataFrame(vin_array, columns=enc.cols + ['seq']).drop(columns=['checkD'])))

        x = dict(zip(RdF_2e.classes_, pred_variante.tolist()[0]))

        sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
        _tab = _tab + [x for x in sorted_x[:3]]


        delete_var(RdF_2e)
        delete_var(enc_2e)

    if to_save:
        _vin.save()

    i = 0
    for  id, prob in _tab:
        if i >= 3 :
            new_prob = prob * tab[1][1]
        else :
            new_prob = prob * tab[0][1]

        vds = vin[6:8]
        result_vds = list(vds_map.objects.filter(vds=vds))
        result_vds = list(map(lambda x: x.gammme, result_vds))

        trusted = True
        if len(result_vds) > 0 :
            result_vds = list(map(lambda x: x.split(','), result_vds))
            result_vds = reduce(lambda a,b: a.append(b), result_vds)
            result_vds = list(map(lambda x: x.strip(), result_vds))

        model = replaceModel(Variante.objects.get(variante_id=id).modelegen)

        val_bol = False
        for mdl in result_vds:
            if len(mdl.split(' ')) == 2:
                if model != mdl:
                    val_bol = False or val_bol
                else:
                    val_bol = True  or val_bol
            elif len(mdl.split(' ')) == 1:
                if model.split(' ')[0] != mdl:
                    val_bol = False or val_bol
                else:
                    val_bol = True or val_bol

        _vin.variante_pred_set.create(variante_id=id, prob=new_prob, trusted=trusted and val_bol, version='v1' )
        i = i + 1
    all = _vin.variante_pred_set.all().order_by('-prob')

    return all

def delete_var(var):
    var = None
    del var

def decodev2(_in  ):
    memory = set()
    to_save = False
    logging.info( 'Prediction : {}'.format(_in) )

    try:
        _vin = Vin.objects.get(vin=_in)
    except ObjectDoesNotExist:
        _vin = Vin(vin=_in)
        to_save = True

    test = pd.DataFrame(fn_v2(_in), columns=enc_v2.cols + ['seq'])
    encoded = enc_v2.transform(test)
    clus = RdF_v2.predict(encoded)[0]

    target_class = clus
    logging.info('- Clus : {}'.format(clus))
    n = clus

    if to_save:
        _vin.save()

    x = None
    cols = None

    while (n < 35):
        RdF_x = joblib.load(  os.path.join( BASE, 'ml_model/v2/model/model_' + str(target_class) + '_.pkl') )
        enc_x = joblib.load(  os.path.join( BASE, 'ml_model/v2/encoder/enc_clus' + str(target_class) + '_.pkl') )

        try:
            cols = _dict[str(target_class)].split(',')
        except:
            cols = _dict[target_class].split(',')

        niv = RdF_x.predict(enc_x.transform(test)[cols])[0]

        target_class = str(target_class) + '_' + str(niv)

        if target_class == '29_9':
            niv = 349555

        n = niv


        x = dict(zip(RdF_x.classes_, RdF_x.predict_proba(enc_x.transform(test)[cols])[0]))
        logging.info('niv : {} - prob : {}'.format(n, x[n]))

    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)

    for itm in [x for x in sorted_x[:4]]:
        _vin.variante_pred_set.create(variante_id=itm[0], prob=itm[1], trusted=True, version='v2')

    if sorted_x[0][1] < 0.7:
        memory.add(target_class.split('_')[-2])
        recall(_vin, test, target_class, memory)
    all = _vin.variante_pred_set.all().order_by('-prob')

    return all

def replaceModel(modelgen):
    li = modelgen.split(' ')
    if (len(li) == 3):
        return  li[0].capitalize() + ' ' + li[1]
    elif(len(li) == 2):
        return li[0].capitalize()

def recall(vin, test, target, memory):
    vin.variante_pred_set.all().update(trusted=False)
    i = 0
    while i in memory:
        i += 1
    target = '_'.join( target.split('_')[:-2] + [str(i)] )

    RdF_x = joblib.load(  os.path.join( BASE, 'ml_model/v2/model/model_' + target + '_.pkl') )
    enc_x = joblib.load(os.path.join(BASE, 'ml_model/v2/encoder/enc_clus' + target + '_.pkl'))

    cols = _dict[str(target)].split(',')
    x = dict(zip(RdF_x.classes_, RdF_x.predict_proba(enc_x.transform(test)[cols])[0]))
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)

    for itm in [x for x in sorted_x[:4]]:
        vin.variante_pred_set.create(variante_id=itm[0], prob=itm[1], trusted=True, version='v2')

    if sorted_x[0][1] < 0.7:
        if sorted_x[0][1] < 0.7:
            memory.add(target.split('_')[-2])
            recall(vin, test, target, memory)