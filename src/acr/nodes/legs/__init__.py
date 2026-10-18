from acr.nodes.legs.application import leg_application
from acr.nodes.legs.behavioural import leg_behavioural
from acr.nodes.legs.bureau import leg_bureau
from acr.nodes.legs.previous import leg_previous

LEG_NODES = {
    "leg_application": leg_application,
    "leg_bureau": leg_bureau,
    "leg_behavioural": leg_behavioural,
    "leg_previous": leg_previous,
}
