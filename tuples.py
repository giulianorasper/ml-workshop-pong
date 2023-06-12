from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

PongObservationUpdate = namedtuple('PongOvservationUpdate', ('transition', 'state'))