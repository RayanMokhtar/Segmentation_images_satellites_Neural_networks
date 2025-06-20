# Generated by Django 5.2.3 on 2025-06-19 23:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flood_prediction', '0005_passwordresettoken'),
    ]

    operations = [
        migrations.CreateModel(
            name='HistoriquePrediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('region', models.CharField(max_length=100, verbose_name='Nom de la région')),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('date_prediction', models.DateField(verbose_name='Date de la prédiction')),
                ('date_execution', models.DateTimeField(auto_now_add=True, verbose_name="Date d'exécution")),
                ('probabilite', models.FloatField(verbose_name="Probabilité d'inondation (%)")),
                ('niveau_risque', models.CharField(max_length=20, verbose_name='Niveau de risque')),
                ('inondation_prevue', models.BooleanField(default=False, verbose_name='Inondation prévue')),
                ('modele_utilise', models.CharField(default='CNN-LSTM', max_length=50, verbose_name='Modèle utilisé')),
            ],
            options={
                'verbose_name': 'Historique de prédiction',
                'verbose_name_plural': 'Historique des prédictions',
                'ordering': ['-date_execution', 'region'],
                'unique_together': {('region', 'date_prediction')},
            },
        ),
    ]
