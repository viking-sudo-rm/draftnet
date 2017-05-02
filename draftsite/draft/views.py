# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.views.generic.base import View
from django.shortcuts import render

# TODO there's probably a better way to do this with an environment variable
import sys
oldPath = sys.path
sys.path.insert(0,'..')
# from bagHeroes import *
from .models import Hero

sys.path = oldPath

# Create your views here.
def index(request):
	context  = {
		"heroArray": Hero.objects.all()
	}
	return render(request, 'draft/index.html', context)