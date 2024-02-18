# Date:     2024-02-10
# Author:   Massimo Clementi <massimo_clementi@icloud.com>
# Topic:    Define enumerators

from enum import Enum

class MatchClassification(Enum):
    CORRESPONDENT = 1
    OCCLUSION = 2
    NEW_MATCH = 3