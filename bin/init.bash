#! /bin/bash

jupyter serverextension enable --py jupyter_http_over_ws
jupyter nbextension enable --py --sys-prefix widgetsnbextension
