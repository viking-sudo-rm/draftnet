# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Hero(models.Model):
	name = models.CharField(max_length=50)
	id = models.IntegerField(primary_key=True)
	localized_name = models.CharField(max_length=50)
	primary_attr = models.CharField(max_length=10)
	attack_type = models.CharField(max_length=10)

# Whenever we start the server, load the heroes with corrrect indexing from util.py
from .views import APIHero

for hero in APIHero.heroes:
	Hero(name=hero.json['name'].replace("npc_dota_hero_", ""), localized_name = hero.json['localized_name'], primary_attr = hero.json['primary_attr'], attack_type=hero.json['attack_type'], id= hero.json['id']).save()