from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Réinitialise les séquences de la base de données pour les tables spécifiées'

    def add_arguments(self, parser):
        parser.add_argument('tables', nargs='+', type=str, help='Liste des tables dont les séquences doivent être réinitialisées')

    def handle(self, *args, **options):
        tables = options['tables']
        
        with connection.cursor() as cursor:
            for table in tables:
                # Récupère l'ID maximum de la table
                cursor.execute(f"SELECT MAX(id) FROM {table};")
                max_id = cursor.fetchone()[0]
                
                if max_id is None:
                    max_id = 0
                
                # Réinitialise la séquence pour commencer à partir de max_id + 1
                cursor.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), {max_id + 1}, false);")
                
                self.stdout.write(self.style.SUCCESS(f'Séquence réinitialisée pour la table {table} à {max_id + 1}'))
