from __future__ import with_statement
import requests
import os, sys

sys.path.insert(0,'../..')
print(os.getcwd())
import util

# os.environ['DJANGO_SETTINGS_MODULE']='project'
from django.conf import settings
settings.configure()

from models import Hero as DbHero

for hero in util.Hero.heroes:
  DbHero(name=hero.json['name'].replace("npc_dota_hero_", ""), localized_name = hero.json['localized_name'], primary_attr = hero.json['primary_attr'], attack_type=hero.json['attack_type'], id= hero.json['id']).save()