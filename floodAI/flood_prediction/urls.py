from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_page, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profil/', views.profil, name='profil'),
    path('register/', views.register, name='register'),
    path('api/weather/', views.get_weather_data, name='get_weather_data'),
    path('api/location/', views.get_location_info, name='get_location_info'),
    path('api/subscribe-city/', views.subscribe_city, name='subscribe_city'),
    path('api/check-subscription/', views.check_subscription_status, name='check_subscription_status'),
    path('unsubscribe-city/<int:abonnement_id>/', views.unsubscribe_city, name='unsubscribe_city'),
    path('segmentation/', views.segmentation, name='segmentation'),
    path('lstm/', views.lstm, name='lstm'),
    path('send-prediction-email/', views.send_prediction_email, name='send_prediction_email'),
]