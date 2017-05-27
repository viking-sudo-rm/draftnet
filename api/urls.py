from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^predict', views.predict, name='predict'),
    url(r'^heroes', views.heroes, name='heroes'),
    url(r'^models', views.models, name='models')
]