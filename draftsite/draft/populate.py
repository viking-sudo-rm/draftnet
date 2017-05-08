import requests
from models import Hero
import util

import os
os.environ['DJANGO_SETTINGS_MODULE']='project.settings'

for hero in util.Hero.heroes:
  Hero(name=hero.json['name'].replace("npc_dota_hero_", ""), localized_name = hero.json['localized_name'], primary_attr = hero.json['primary_attr'], attack_type = hero.json['attack_type'], id= hero.json['id']).save()