
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
from agents1.BW4TBaselineAgent import BaseLineAgent, Phase

import json

# World Knowledge that out agent acquires throughout the run
class WorldKnowledge:
    opened_doors = [] # The list of doors opened (by anyone) on the map
    visited_rooms = {} # The dictionary of visited (by us) rooms with associated block locations inside
    agent_speeds = {} # The dictionary of agents and their corresponding speeds

    def __init__(self) -> None:
        pass

    def door_opened(self, door_name):
        self.opened_doors.append(door_name)

    def room_visited(self, room_name, blocks):
        self.visited_rooms[room_name] = blocks


class StrongAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._currentlyCarryingTwo = -1
        self._checkedNearbyBlocks = False

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
                            print(self._nearbyGoalBlocksStored)
                        else:
                            if objCarryId not in self._nearbyGoalBlocksStored[index_obj]:
                                print(self._nearbyGoalBlocksStored)
                                self._nearbyGoalBlocksStored[index_obj].append(
                                    objCarryId)

            if Phase.PLAN_PATH_TO_CLOSED_DOOR == self._phase:
                self._navigator.reset_full()

                doors = [door for door in state.values() if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]

                if len(doors) == 0 and not self._checkedNearbyBlocks:
                    self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                elif len(doors) == 0 and self._checkedNearbyBlocks:

                    doors = [door for door in state.values() if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

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

                else:
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
                roomObjects = [x for x in roomObjects if x['is_collectable'] == True]

                for obj in roomObjects:
                    result = self._checkIfDesiredBlock(obj)

                    if result[0]: # if desired block found
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([result[1]])

                        if self._currentlyCarrying == -1: self._currentlyCarrying = result[4]
                        elif self._currentlyCarrying != -1 and self._currentlyCarryingTwo == -1: self._currentlyCarryingTwo = result[4]
                        else: return print('we did something wrong')

                        if result[3]: # if currently desired block found
                            self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT
                        elif not result[3] and self._currentlyCarryingTwo != -1:
                            self._sendMessage('Spotted goal object ' + result[2] + ' at ' + self._door['room_name'] + ", Index: " + str(result[4]), agent_name)
                            self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE
                        else:
                            self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR

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

                self._sendMessage(
                    'Found currently desired object ' + str(self._currentlyCarrying), agent_name)

                if self._currentlyWantedBlock < len(self._goalBlockCharacteristics) - 1:
                    self._currentlyWantedBlock += 1

                if self._currentlyCarryingTwo != -1: objCarryId = state[self.agent_id]['is_carrying'][1]['obj_id']
                else: objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']

                self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY

                if self._currentlyCarryingTwo != -1: self._currentlyCarryingTwo = -1
                else: self._currentlyCarrying = -1

                return DropObject.__name__, {'object_id': objCarryId}

            if Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}

                if self._currentlyCarryingTwo != -1: objCarryId = state[self.agent_id]['is_carrying'][1]['obj_id']
                else: objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']

                objectIndex = self._currentlyCarryingTwo if self._currentlyCarryingTwo != -1 else self._currentlyCarrying
                visualizationObj = str(
                    state[self.agent_id]['is_carrying'][0]['visualization'])
                self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                self._sendMessage(
                    'Stored nearby the goal object ' + '{' + objCarryId + "}" + "with " + "[" + visualizationObj + "]" + ", Index: " + str(self._currentlyCarrying), agent_name)

                self._checkGoalBlocksPlacedNearby[objectIndex] = True
                if objectIndex not in self._nearbyGoalBlocksStored:
                    list_obj = []
                    list_obj.append((objCarryId, visualizationObj))
                    self._nearbyGoalBlocksStored[objectIndex] = list_obj
                    print(self._nearbyGoalBlocksStored)

                else:
                    self._nearbyGoalBlocksStored[objectIndex].append(objCarryId)
                    print(self._nearbyGoalBlocksStored)

                if self._currentlyCarryingTwo != -1: self._currentlyCarryingTwo = -1
                else: self._currentlyCarrying = -1

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

                for storedBlockID in self._nearbyGoalBlocksStored[self._currentlyWantedBlock]:
                    desiredBlock = self._goalBlockCharacteristics[
                        self._currentlyWantedBlock]['visualization']
                    # THE CHARACTERISTICS OF THE STORED BLOCK SHOULD BE GET HERE
                    storedBlock = storedBlockID[1]
                    storedSize = float(storedBlock[storedBlock.find(
                        "'size': ")+8:storedBlock.find(",")])
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
                if self._currentlyWantedBlock == self._currentlyCarrying:
                    self._navigator.reset_full()
                    currentlyDesiredDropzone = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']

                    dropzoneLocation = currentlyDesiredDropzone[0], currentlyDesiredDropzone[1]
                    self._navigator.add_waypoints([dropzoneLocation])

                    self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT

                elif self._currentlyCarrying != -1:
                    self._navigator.reset_full()
                    notCurrentlyDesiredDropzone = self._goalBlockCharacteristics[self._currentlyCarrying]['location']

                    dropzoneLocation = notCurrentlyDesiredDropzone[0] + 3, notCurrentlyDesiredDropzone[1]
                    self._navigator.add_waypoints([dropzoneLocation])

                    self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE

                elif self._currentlyWantedBlock in self._nearbyGoalBlocksStored:
                    self._navigator.reset_full()
                    print('OH YEAH')

                    block_location = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']
                    block_location = block_location[0] + 3, block_location[1]
                    self._navigator.add_waypoints([block_location])

                    self._phase = Phase.GRAB_DESIRED_OBJECT_NEARBY

                else:
                    self._phase = Phase.PLAN_PATH_TO_CLOSED_DOOR
