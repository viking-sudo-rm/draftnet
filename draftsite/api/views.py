# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from django.http import HttpResponse, JsonResponse
from django.views.generic.base import View
from django.views.decorators.csrf import csrf_exempt
import json, os, sys

# TODO there's probably a better way to do this with an environment variable
sys.path.insert(0,'..')
from bagHeroes import *
# sys.path.pop(0)

MODEL = "../results/bag-100-1000000-0.01-50.ckpt"

# session = tf.get_default_session()
with session.as_default():
    saver = tf.train.Saver()
    saver.restore(session, MODEL)

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

	team0 = Team.fromJSON(args["team0"])
	team1 = Team.fromJSON(args["team1"])
	isPick = args["isPick"]

	if not team0 or not team1:
		return JsonResponse(None, safe=False)

	# TODO this session should be always loaded?

	context = getContext(team0, team1, isPick)
	distribution = getDistribution(context, session)
	suggestions = getSuggestions(distribution, getNotAllowed(context))

	return JsonResponse({	"distribution": [float(d) for d in distribution],
							"suggestions": suggestions	})