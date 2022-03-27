
from operator import index
from re import L
from typing import final, List, Dict, Final
import enum
import random
from cv2 import phase
from numpy import place

from sqlalchemy import null
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message


class Phase(enum.Enum):
    PLAN_PATH_TO_CLOSED_DOOR = 1,
    FOLLOW_PATH_TO_CLOSED_DOOR = 2,
    OPEN_DOOR = 3,

    GRAB_OBJECT = 4,
    DROP_OBJECT = 5,
    SCAN_ROOM = 6,
    SEARCH_AND_FIND_GOAL_BLOCK = 7,
    ENTER_THE_ROOM = 8,
    PLAN_TO_DROP_ZONE = 10.
    FILLING_BLOCK_IN_DROP_ZONE = 11,
    PLAN_TO_NEXT_TO_GOAL_BLOCK = 12,
    GRAB_OBJECT_AT_NEXT_TO_GOAL_BLOCK = 13,
    CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY = 14,

    PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT = 15,
    PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE = 16,
    GRAB_DESIRED_OBJECT_NEARBY = 17


class BaseLineAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
        self._teamMembers = []

        self._trustBeliefs = {}
        self._goalBlockCharacteristics = []
        self._checkGoalBlocksPlacedNearby = []
        self._nearbyGoalBlocksStored = {}
        self._alreadyPutInDropZone = set()

        self._countPut = 0
        self._goingCurrLocation = None
        self._currentlyWantedBlock = 0
        self._currentlyCarrying = -1

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        print(self.agent_id)
        print("="*10)
        print(state)
        return state

    def _checkIfDesiredBlock(self, curr_obj):

        c_size = curr_obj['visualization']['size']
        c_shape = curr_obj['visualization']['shape']
        c_colour = curr_obj['visualization']['colour']

        for index, goal in enumerate(self._goalBlockCharacteristics):

            g_size = goal['visualization']['size']
            g_shape = goal['visualization']['shape']
            g_colour = goal['visualization']['colour']

            if(c_size == g_size and c_shape == g_shape and c_colour == g_colour):
                obj_description = "{size : " + str(g_size) + ", " + \
                    "shape : " + str(g_shape) + ", " + \
                    "colour : " + g_colour + ", " + \
                    "order : " + str(index) + "}"
                result = self._currentlyDesiredOrNot(index, goal)
                return (True, result[0], obj_description, result[1], index)

        return (False, None)

    def _currentlyDesiredOrNot(self, index, goal_block):
        if index == self._currentlyWantedBlock and index not in self._alreadyPutInDropZone:
            self._alreadyPutInDropZone.add(index)
            return goal_block['location'], True
        elif index - 1 not in self._alreadyPutInDropZone:
            goal_loc = goal_block['location']
            return (goal_loc[0] + 3, goal_loc[1]), False
        else:
            self._alreadyPutInDropZone.add(index)
            return goal_block['location'], True

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    ############################################################################################################
    ############################### Decide on the action based on trust belief #################################
    ############################################################################################################

    def decide_on_bw4t_action(self, state: State):
        '''
            • Moving to [room_name] 4
            – e.g., Moving to room_4

            • Opening door of [room_name]
            – e.g., Opening door of room_4 • Searching through [room_name]
            – e.g., Searching through room_4

            • Found goal block [block_visualization] at location [location]
            – e.g., Found goal block {"size": 0.5, "shape": 1, "colour": "#0008ff"} at location (8, 8)

            • Picking up goal block [block_visualization] at location [location]
            – e.g., Picking up goal block {"size": 0.5, "shape": 1, "colour": "#0008ff"}
            at location (8, 8)

            • Dropped goal block [block_visualization] at drop location [location]
            – e.g., Dropped goal block {"size": 0.5, "shape": 1, "colour": "#0008ff"} at drop location (11, 23)
        '''

        # Get information about goal-blocks and their location
        if len(self._goalBlockCharacteristics) == 0:
            dropZones = state.get_with_property('drop_zone_nr')
            self._goalBlockCharacteristics = [
                x for x in dropZones if x['is_goal_block'] == True]
            print(self._goalBlockCharacteristics)
            print("\n\n")

        # Initialize a list that checks whether the goal block is placed nearby
        if len(self._checkGoalBlocksPlacedNearby) == 0:
            self._checkGoalBlocksPlacedNearby = [
                False for x in range(len(self._goalBlockCharacteristics))]
            print(self._checkGoalBlocksPlacedNearby)

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)
        # Update trust beliefs for team members
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:

            closedDoors = [door for door in state.values()
                           if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
            received = self._processMessages(self._teamMembers)

            for member in received.keys():
                for message in received[member]:
                    if 'Found currently' in message:
                        self._alreadyPutInDropZone.add(
                            int(message[len(message) - 1]))
                        self._currentlyWantedBlock = int(
                            message[len(message) - 1]) + 1
                    if 'Stored nearby' in message:
                        objCarryId = message[message.find(
                            "{")+1:message.find("}")]
                        index_obj = int(message[len(message) - 1])

                        self._checkGoalBlocksPlacedNearby[index_obj] = True
                        if index_obj not in self._nearbyGoalBlocksStored:
                            list_obj = []
                            list_obj.append(objCarryId)
                            self._nearbyGoalBlocksStored[index_obj] = list_obj
                            print(self._nearbyGoalBlocksStored)
                        else:
                            if objCarryId not in self._nearbyGoalBlocksStored[index_obj]:
                                print(self._nearbyGoalBlocksStored)
                                self._nearbyGoalBlocksStored[index_obj].append(
                                    objCarryId)

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                closedDoors = [door for door in state.values()
                               if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
                if len(closedDoors) == 0:
                    return None, {}
                # Randomly pick a closed door
                self._door = random.choice(closedDoors)
                doorLoc = self._door['location']

                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1]+1

                # Send message of current action
                self._sendMessage('Moving to door of ' +
                                  self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_CLOSED_DOOR

            if Phase.FOLLOW_PATH_TO_CLOSED_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_THE_ROOM
                # Open door
                self._sendMessage('Opening door of ' +
                                  self._door['room_name'], agent_name)

                enterLoc = self._door['location']
                enterLoc = enterLoc[0], enterLoc[1] - 1
                self._navigator.add_waypoints([enterLoc])

                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.ENTER_THE_ROOM == self._phase:
                self._state_tracker.update(state)

                # Enter the room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}

                self._sendMessage('Entering the ' +
                                  self._door['room_name'], agent_name)
                self._phase = Phase.SCAN_ROOM

            if Phase.SCAN_ROOM == self._phase:
                self._navigator.reset_full()
                roomInfo = state.get_room_objects(self._door['room_name'])
                roomArea = [area['location'] for area in roomInfo if area['name']
                            == self._door['room_name'] + "_area"]
                self._navigator.add_waypoints(roomArea)

                self._sendMessage('Scanning ' +
                                  self._door['room_name'], agent_name)
                self._phase = Phase.SEARCH_AND_FIND_GOAL_BLOCK

            if Phase.SEARCH_AND_FIND_GOAL_BLOCK == self._phase:

                self._state_tracker.update(state)
                action = self._navigator.get_move_action(
                    self._state_tracker)

                roomObjects = state.get_closest_with_property('is_goal_block')
                roomObjects = [
                    x for x in roomObjects if x['is_collectable'] == True]
                for obj in roomObjects:
                    result = self._checkIfDesiredBlock(obj)
                    if result[0]:
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([result[1]])
                        self._currentlyCarrying = result[4]
                        if result[3]:
                            self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT
                            self._sendMessage(
                                'Found currently desired object ' + str(result[4]), agent_name)
                        else:
                            self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE
                            self._sendMessage('Spotted goal object ' + result[2] + ' at ' +
                                              self._door['room_name'] + ", Index: " + str(result[4]), agent_name)

                        return GrabObject.__name__, {'object_id': obj['obj_id']}

                if action != None:
                    return action, {}

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}

                if self._currentlyWantedBlock < len(self._goalBlockCharacteristics) - 1:
                    self._currentlyWantedBlock += 1

                objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                self._currentlyCarrying = -1

                return DropObject.__name__, {'object_id': objCarryId}

            if Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}

                objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                self._sendMessage(
                    'Stored nearby the goal object ' + '{' + objCarryId + "}" + ", Index: " + str(self._currentlyCarrying), agent_name)

                self._checkGoalBlocksPlacedNearby[self._currentlyCarrying] = True
                if self._currentlyCarrying not in self._nearbyGoalBlocksStored:
                    list_obj = []
                    list_obj.append(objCarryId)
                    self._nearbyGoalBlocksStored[self._currentlyCarrying] = list_obj
                    print(self._nearbyGoalBlocksStored)
                else:
                    self._nearbyGoalBlocksStored[self._currentlyCarrying].append(
                        objCarryId)
                    print(self._nearbyGoalBlocksStored)

                self._currentlyCarrying = -1
                print(self._checkGoalBlocksPlacedNearby)

                return DropObject.__name__, {'object_id': objCarryId}

            if Phase.GRAB_DESIRED_OBJECT_NEARBY == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}

                block_location = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']
                self._navigator.reset_full()
                self._navigator.add_waypoints([block_location])

                obj = self._nearbyGoalBlocksStored[self._currentlyWantedBlock][0]
                self._nearbyGoalBlocksStored[self._currentlyWantedBlock].pop(0)
                self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT

                return GrabObject.__name__, {'object_id': obj}

            if Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY == self._phase:
                if self._currentlyWantedBlock in self._nearbyGoalBlocksStored:
                    self._navigator.reset_full()
                    print('OH YEAH')

                    block_location = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']
                    block_location = block_location[0] + 3, block_location[1]
                    self._navigator.add_waypoints([block_location])

                    self._phase = Phase.GRAB_DESIRED_OBJECT_NEARBY

                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

    ############################################################################################################
    ########################################## Our trust belief system #########################################
    ############################################################################################################

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member,
        for example based on the received messages. You can change the default value to your preference

        Statistical aggregation for now
        '''

        default = 0.4

        for member in received.keys():
            if member not in self._trustBeliefs:
                self._trustBeliefs[member] = default

        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    self._trustBeliefs[member] -= 0.1
                    break

        '''

        Lazy
        Colorblind

        Strong
        Liar

        '''
