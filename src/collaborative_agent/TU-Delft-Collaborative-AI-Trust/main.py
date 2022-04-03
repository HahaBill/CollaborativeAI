from bw4t.BW4TWorld import BW4TWorld
from bw4t.statistics import Statistics
from agents1.BW4TBaselineAgent import BaseLineAgent
from agents1.BW4TStrongAgent import StrongAgent
from agents1.BW4TBlindAgent import BlindAgent
from agents1.BW4TLazyAgent import LazyAgent
from agents1.BW4TLiarAgent import LiarAgent
from agents1.BW4THuman import Human


"""
This runs a single session. You have to log in on localhost:3000 and
press the start button in god mode to start the session.
"""

if __name__ == "__main__":
    agents = [
        {'name': 'baseAgent0', 'botclass': BaseLineAgent, 'settings': {}},
        {'name': 'baseAgent1', 'botclass': BaseLineAgent, 'settings': {}},
        #{'name': 'liarAgent', 'botclass': LiarAgent, 'settings': {}},
        {'name': 'baseAgent2', 'botclass': BaseLineAgent, 'settings': {}},
        # {'name': 'baseAgent3', 'botclass': BaseLineAgent, 'settings': {}},
        #{'name': 'strongAgent', 'botclass': StrongAgent, 'settings': {}},
        #{'name': 'blindAgent', 'botclass': BlindAgent, 'settings': {}},
        #{'name': 'secondBlind', 'botclass': BlindAgent, 'settings': {}},c
        #{'name': 'thirdBlind', 'botclass': BlindAgent, 'settings': {}},
        # {'name': 'lazyAgent', 'botclass': LazyAgent, 'settings': {}}
    ]

    print("Started world...")
    world = BW4TWorld(agents).run()
    print("DONE!")
    print(Statistics(world.getLogger().getFileName()))
