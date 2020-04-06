#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Pete Bunting'
SITENAME = 'Remote Sensing . Info'
SITEURL = ''

THEME = 'theme'

PATH = 'content'

TIMEZONE = 'Europe/London'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('RSGISLib', 'https://www.rsgislib.org'),
         ('ARCSI', 'https://arcsi.remotesensing.info'),
         ('EODataDown', 'https://eodatadown.remotesensing.info'),
         ('KEALib', 'https://kealib.org'),
         ('SPDlib', 'https://www.spdlib.org'),
         ('pylidar', 'https://pylidar.org'),
         ('TuiView', 'https://tuiview.org'),)

# Social widget
SOCIAL = (('Bitbucket', 'https://bitbucket.org/petebunting/'),
         ('GitHub', 'https://github.com/petebunting'),)

DEFAULT_PAGINATION = False

PLUGIN_PATHS=['./plugins']
PLUGINS = ['render_math']


# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
