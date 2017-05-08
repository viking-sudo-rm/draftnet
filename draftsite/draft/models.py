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