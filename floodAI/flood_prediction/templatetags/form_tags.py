from django import template

register = template.Library()

@register.filter(name='add_class')
def add_class(value, arg):
    """
    Ajoute une classe CSS au champ de formulaire
    Usage: {{ form.field|add_class:"form-control" }}
    """
    css_classes = value.field.widget.attrs.get('class', '')
    
    # Si la classe est déjà présente, ne pas l'ajouter à nouveau
    if arg not in css_classes:
        css_classes = f"{css_classes} {arg}".strip()
    
    return value.as_widget(attrs={'class': css_classes})
