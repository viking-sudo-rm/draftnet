import requests
from models import Hero

import os
os.environ['DJANGO_SETTINGS_MODULE']='project.settings'

heroes = requests.get("https://api.opendota.com/api/heroes").json()

for hero in heroes:
  Hero(name=hero['name'].replace("npc_dota_hero_", ""), localized_name = hero['localized_name'], primary_attr = hero['primary_attr'], attack_type = hero['attack_type']).save()