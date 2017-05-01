from django.conf.urls import url

from .views import Homepage


urlpatterns = [
    url(r'^$', Homepage.as_view(), name='index'),
]