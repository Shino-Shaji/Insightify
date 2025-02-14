import markdown as md
from django import template
from django.conf import settings
from django.template.defaultfilters import stringfilter


register = template.Library()

@register.filter()
@stringfilter
def markdown(value):
    extensions = getattr(settings, 'MARKDOWN_EXTENSIONS', [])
    return md.markdown(value, extensions=extensions)