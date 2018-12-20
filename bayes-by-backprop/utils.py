from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
