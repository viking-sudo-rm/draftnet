# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.views.generic.base import View
from django.shortcuts import render

# Create your views here.
def index(request):
	return render(request, 'draft/index.html')