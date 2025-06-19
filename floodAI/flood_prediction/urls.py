from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_page, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profil/', views.profil, name='profil'),
    path('register/', views.register, name='register'),
    path('verify-email/<uuid:token>/', views.verify_email, name='verify_email'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),
    path('reset-password/<uuid:token>/', views.reset_password, name='reset_password'),
    path('api/weather/', views.get_weather_data, name='get_weather_data'),
    path('api/location/', views.get_location_info, name='get_location_info'),
    path('api/subscribe-city/', views.subscribe_city, name='subscribe_city'),
    path('api/check-subscription/', views.check_subscription_status, name='check_subscription_status'),
    path('unsubscribe-city/<int:abonnement_id>/', views.unsubscribe_city, name='unsubscribe_city'),
    path('segmentation/', views.segmentation, name='segmentation'),
    path('lstm/', views.lstm, name='lstm'),
    path('send-city-alert/', views.send_city_alert, name='send_city_alert'),
]