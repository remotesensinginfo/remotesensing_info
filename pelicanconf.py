#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Pete Bunting'
SITENAME = 'Remote Sensing . Info'
SITEURL = 'http://remotesensing.info'

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
LINKS = (('RSGISLib', 'http://www.rsgislib.org'),
         ('ARCSI', 'http://remotesensing.info/arcsi'),
         ('EODataDown', 'http://remotesensing.info/eodatadown'),
         ('KEALib', 'http://kealib.org'),
         ('SPDlib', 'http://www.spdlib.org'),
         ('pylidar', 'http://pylidar.org'),
         ('TuiView', 'http://tuiview.org'),)

# Social widget
SOCIAL = (('GitHub', 'https://github.com/petebunting'),
          ('YouTube', 'https://www.youtube.com/drpetebunting'),)

DEFAULT_PAGINATION = False

PLUGIN_PATHS=['./plugins']
PLUGINS = ['render_math']


# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
