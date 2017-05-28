# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from django.http import HttpResponse, JsonResponse
from django.views.generic.base import View
from django.views.decorators.csrf import csrf_exempt
from django.core import serializers
import json, sys

from draftnet import *
from .models import Hero

MODEL = "results/pub-7.06-3809.ckpt"
# MODEL = "results/pro-smaller-7.00.ckpt"
# MODEL = "results/7.06-136.json"

# with session.as_default():
#     saver = tf.train.Saver()
#     saver.restore(session, MODEL)

@csrf_exempt # I don't think this causes security issues, but maybe?

def predict(request):
	try:
		args = json.loads(request.body)
	except:
		return JsonResponse(None, safe=False)

	if type(args) != dict:
		return JsonResponse(None, safe=False)

	if "team0" not in args or "team1" not in args or "isPick" not in args or type(args["isPick"]) != bool:
		return JsonResponse(None, safe=False)

	if "model" not in args or args["model"] not in sessions:
		return JsonResponse(None, safe=False)

	team0 = Team.fromJSON(args["team0"])
	team1 = Team.fromJSON(args["team1"])
	isPick = args["isPick"]

	if not team0 or not team1:
		return JsonResponse(None, safe=False)

	next_context = getContext(team0, team1, isPick, args["side"])
	session = sessions[args["model"]]
	distribution = getDistribution(context, session, graph)
	suggestions = getSuggestions(distribution, getNotAllowed(context))
	if isPick:
		if args["side"] == 1:
			advantage_context = team1.pickVector + team0.pickVector
		else:
			advantage_context = team0.pickVector + team1.pickVector
		advantage = getDistribution(context, session, graph) # make graph WinGraph

	return JsonResponse({	"distribution": [float(d) for d in distribution],
							"suggestions": suggestions,
							"advantage": advantage	})

def heroes(request):
	heroList = json.loads(serializers.serialize('json', Hero.objects.all()))
	result = []
	for hero in heroList:
		obj = hero['fields']
		obj['id'] = hero['pk']
		result.append(obj)
	return JsonResponse(result, safe=False)

def models(request):
	return JsonResponse(sessionNames, safe=False)