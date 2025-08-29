from enum import Enum
class Direction(Enum):
        RIGHT=1
        LEFT=2
        UP=3
        DOWN=4
        NoDIR=5
        AICALL=6


class Flags(Enum):
    EmptyGround=0
    HitItselfOrBorder=1
    HitPLAYERSNAKE2=3
    HitAISNAKE=2
    AteNormalFruit=4
    AteBonusFruit=5
    AtePoisonusFruit=6
    HitPLAYERSNAKE1=7
    AteNormalFruit2=8
    AteNormalFruit3=9
    AteNormalFruit4=10
    AteNormalFruit5=11
