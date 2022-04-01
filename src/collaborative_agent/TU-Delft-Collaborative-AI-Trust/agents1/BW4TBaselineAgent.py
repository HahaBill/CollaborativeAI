
from operator import index
from re import L
from typing import final, List, Dict, Final
import enum
import random
# from cv2 import phase
from numpy import place

from sqlalchemy import null
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message

import json

# World Knowledge that out agent acquires throughout the run


class WorldKnowledge:
    opened_doors = []  # The list of doors opened (by anyone) on the map
    # The dictionary of visited (by us) rooms with associated block locations inside
    visited_rooms = {}
    agent_speeds = {}  # The dictionary of agents and their corresponding speeds

    def __init__(self) -> None:
        pass

    def door_opened(self, door_name):
        self.opened_doors.append(door_name)

    def room_visited(self, room_name, blocks):
        self.visited_rooms[room_name] = blocks


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

        ## Fields specific to the Trust Model
        self._defaultBeliefAssignment = True
        self._defaultTrustValue = 0.6 # Somewhat optimistic to start with
        self._closedRooms = {} # The list is updated every tick
        self._visitedRooms = {} # Blocks with locations are added every time a new one is noticed in a room

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

        # Record the list of currently closed doors
        self._closedRooms = [door['room_name'] for door in state.values(
                ) if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']] 
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:

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
                        visualizationObj = str(message[message.find(
                            "[")+1:message.find("]")])
                        index_obj = int(message[len(message) - 1])

                        self._checkGoalBlocksPlacedNearby[index_obj] = True
                        if index_obj not in self._nearbyGoalBlocksStored:
                            list_obj = []
                            list_obj.append((objCarryId, visualizationObj))
                            self._nearbyGoalBlocksStored[index_obj] = list_obj
                            # print(self._nearbyGoalBlocksStored)
                        else:
                            if objCarryId not in self._nearbyGoalBlocksStored[index_obj]:
                                # print(self._nearbyGoalBlocksStored)
                                self._nearbyGoalBlocksStored[index_obj].append(
                                    (objCarryId, visualizationObj))

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                doors = [door for door in state.values(
                ) if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
                if len(doors) == 0:
                    doors = [door for door in state.values(
                    ) if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

                # Randomly pick a closed door
                self._door = random.choice(doors)
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

                room_name = self._door['room_name']
                self.visit_new_room(room_name)

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
                action = self._navigator.get_move_action(self._state_tracker)

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
                        else:
                            self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE
                            self._sendMessage('Spotted goal object ' + result[2] + ' at ' +
                                              self._door['room_name'] + ", Index: " + str(result[4]), agent_name)

                        return GrabObject.__name__, {'object_id': obj['obj_id']}
                    
                    room_name = self._door['room_name'] 
                    self.discover_block_in_visited_room(obj, room_name)

                if action != None:
                    return action, {}

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

            if Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}

                self._sendMessage(
                    'Found currently desired object ' + str(self._currentlyCarrying), agent_name)

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
                visualizationObj = str(
                    state[self.agent_id]['is_carrying'][0]['visualization'])
                self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                self._sendMessage(
                    'Stored nearby the goal object ' + '{' + objCarryId + "}" + "with " + "[" + visualizationObj + "]" + ", Index: " + str(self._currentlyCarrying), agent_name)

                self._checkGoalBlocksPlacedNearby[self._currentlyCarrying] = True
                if self._currentlyCarrying not in self._nearbyGoalBlocksStored:
                    list_obj = []
                    list_obj.append((objCarryId, visualizationObj))
                    self._nearbyGoalBlocksStored[self._currentlyCarrying] = list_obj
                    # print(self._nearbyGoalBlocksStored)
                else:
                    self._nearbyGoalBlocksStored[self._currentlyCarrying].append(
                        objCarryId)
                    # print(self._nearbyGoalBlocksStored)

                self._currentlyCarrying = -1
                # print(self._checkGoalBlocksPlacedNearby)

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

                for storedBlockID in self._nearbyGoalBlocksStored[self._currentlyWantedBlock]:
                    desiredBlock = self._goalBlockCharacteristics[
                        self._currentlyWantedBlock]['visualization']
                    # THE CHARACTERISTICS OF THE STORED BLOCK SHOULD BE GET HERE
                    storedBlock = storedBlockID[1]
                    storedSize = float(storedBlock[storedBlock.find(
                        "'size': ")+8:storedBlock.find(",")])
                    print("AFTER SIZE AND AGAIN THANKS BLIND !! ", storedBlock)
                    storedShape = int(storedBlock[storedBlock.find(
                        "'shape': ")+9:storedBlock.find(", 'co")])
                    storedColour = storedBlock[storedBlock.find(
                        "'colour': ")+11:storedBlock.find(", 'de") - 1]

                    if storedShape == int(desiredBlock['shape']) and storedSize == float(desiredBlock['size']) and storedColour == desiredBlock['colour']:
                        self._nearbyGoalBlocksStored[self._currentlyWantedBlock].remove(
                            storedBlockID)
                        self._currentlyCarrying = self._currentlyWantedBlock
                        self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT
                        return GrabObject.__name__, {'object_id': storedBlockID[0]}

                self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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

    def update_mem(self):
        '''
        Update the records of trust for this agent in the memory file
        '''
        trustor = self.agent_id  # Get the name (ID) of the trustor
        mem_entry = {trustor: self._trustBeliefs}
        # !! The path starts at CAI project. I think
        # we should change the project so that only
        # collaborative agent is there, without the rest of it.
        # Perhaps, create a seperate repository, for clarity.
        with open('./src/collaborative_agent/TU-Delft-Collaborative-AI-Trust/agents1/Memory.json', 'w') as outfile:
            json.dump(mem_entry, outfile)

    def visit_new_room(self, room_name):
        # Add a new room entry to the dict of visisted rooms (unless it exists already)
        if (room_name not in self._visitedRooms.keys()):
            self._visitedRooms[room_name] = []
        

    def discover_block_in_visited_room(self, block_obj, room_name):
        if (room_name not in self._visitedRooms.keys()):
            raise Exception("Discovered a block in a not yet visited room.\n")

        self._visitedRooms[room_name].append(block_obj)
        print("Discovered a ", block_obj['visualization'], " in room ", room_name)

    def extract_from_message_found_block(self, message):
            split_1 = message.split("{")[1]
            split_2 = split_1.split("}")

            split_block = split_2[0].replace(",", ":").split(": ")
            block_size = split_block[1]
            block_shape = split_block[3]
            block_color = split_block[5]

            split_location = split_2[1].split()
            x = split_location[2].replace("(", "").replace(",", "")
            y = split_location[3].replace(")", "")

            return block_size, block_shape, block_color, x, y

    def direct_exp(self, trustee, messages):
        curr_trust = self._trustBeliefs[trustee]

        for message in messages:
            # Definitive evidence #
            if ('Opening door' in message):
                message_words = message.split()
                room_name = message_words[3]  # Get room name from the message
                if (room_name not in self._closedRooms):
                    curr_trust = 0.0 # If the room is not open, then agent is lying

            if ('Found goal block' in message):
                message_data = self.extract_from_message_found_block(message)
                found_block_vis = message_data[0:3:] # Get block visualization from the message
                found_block_location = message_data[3::] # Get block location from the message
                found = False
                for block_obj in self._visitedRooms.values():
                    block_vis = block_obj["vizualization"]
                    block_location = block_obj["location"]
                    if (block_vis == found_block_vis and found_block_location == block_location or found_block_location == block_location):
                        found = True
                        curr_trust = 0.0
                        break
            
            # if ('?' in message):
            #     agent_start = ... # Get the starting point
            #     agent_dest = ... # Get the destination
                

            # trustee_speed = self.world_knowledge.agent_speeds[trustee]
            # time_passed = current_time - time_of_message
            # if (|trustee_location - meeting_point| > trustee_speed * time_passed):
            #     curr_trust = 0.0

            # Partial evidence #
            # sus_dist = ...
            # if (|trustee_location - meeting_point| - trustee_speed * time_passed > sus_dist):
            #     curr_trust = 0.0

        self._trustBeliefs[trustee] = curr_trust

    def image(self, trustees, all_messages):
        for trustee in trustees:
            self.direct_exp(trustee, all_messages[trustee])
            # self.comm_exp(trustee, all_messages)
            # reputation. How to implement?

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member,
        for example based on the received messages. You can change the default value to your preference

        Statistical aggregation for now
        '''

        # Assign the default trust value to all other agents
        if (self._defaultBeliefAssignment):
            for member in received.keys():
                if member not in self._trustBeliefs:
                    self._trustBeliefs[member] = self._defaultTrustValue

        # return
        # Generate the trust values for every trustee given their messages
        self.image(received.keys(), received)
        # Save the result trust beliefs in the memory file
        #self.update_mem()

        '''

        Lazy
        Colorblind

        Strong
        Liar

        '''
