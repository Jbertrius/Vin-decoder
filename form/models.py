from django.db import models

# Create your models here.


class Vin(models.Model):
    vin = models.CharField(max_length=17)

    def __str__(self):
        return "%s" % (self.vin)



class Variante(models.Model):
    variante_id = models.IntegerField(primary_key=True)

    marque = models.CharField(max_length=40, null=True)
    modelegen = models.CharField(max_length=40, null=True)
    cylindreelit = models.FloatField(null=True)
    alimentation = models.CharField(max_length=25, null=True)
    capacitecarter = models.FloatField(null=True)
    carosserie = models.CharField(max_length=40, null=True)
    cylindresnbr =  models.IntegerField(null=True)
    energie = models.CharField(max_length=50, null=True)
    phase = models.IntegerField(null=True)
    portesnbr = models.IntegerField(null=True)
    puissancecom = models.IntegerField(null=True)
    typeboitevitesses = models.CharField(max_length=40, null=True)
    vitessesbte = models.CharField(max_length=50, null=True)
    injection = models.CharField(max_length=10, null=True)


class Variante_pred(models.Model):
    variante = models.ForeignKey(Variante,models.CASCADE)
    prob = models.FloatField(null=True)
    vin = models.ForeignKey(Vin, on_delete=models.CASCADE)
    trusted = models.BooleanField(default=True)
    version = models.CharField(max_length=5, null=True)

    def __str__(self):
        return " {} {} ".format(self.variante_id, self.prob)


class vds_map(models.Model):
    vds = models.CharField(max_length=2)
    gammme = models.CharField(max_length=110)


class plant_map(models.Model):
    plant = models.CharField(max_length=1)
    plant_name = models.CharField(max_length=64)