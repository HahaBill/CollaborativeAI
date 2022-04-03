from cmath import phase
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


"""
The LazyAgent

Less willing to use energy and thus sometimes stops what it is doing. This agent does not complete
the action they say they will do 50% of the time, and start another task/action instead (and communicate this new action,
therefore they do not lie). For example, this agent may stop searching room X after a few moves, and move to another room instead.

Additional features from the BaselineAgent :


"""


class LazyAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._will_quit = False
        self._number_of_steps_to_take = 0

        self.decide_if_quit(0)

    ############################################################################################################
    ############################### Decide on the action based on trust belief #################################
    ############################################################################################################

    def check_if_quit(self):
        return True if self._will_quit == True and self._number_of_steps_to_take <= 0 else False

    def decide_if_quit(self, max_steps):
        self._will_quit = bool(random.randint(0, 1))
        if self._will_quit:
            self._number_of_steps_to_take = random.randint(0, max_steps)

    def decide_on_bw4t_action(self, state: State):

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
                if (self._defaultAgentsInRooms):
                    self._agents_in_rooms[member] = None
        self._defaultAgentsInRooms = False
        # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Update trust beliefs for team members
        self._valid_rooms = [door['room_name'] for door in self._state.values(
        ) if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
        # Record the list of currently closed doors
        self._closedRooms = [door['room_name'] for door in state.values(
        ) if 'class_inheritance' in door and 'Door' in door['class_inheritance'] and not door['is_open']]
        self._trustBlief(self._teamMembers, receivedMessages)

        while True:

            received = self._processMessages(self._teamMembers)

            for member in received.keys():
                for message in received[member]:
                    if 'Put currently' in message:
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

            """
            Planning a path to a randomly chosen door

            """
            if Phase.PLAN_PATH_TO_DOOR == self._phase:
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

                self.decide_if_quit(0)

                if not self.check_if_quit():
                    self._navigator.add_waypoints([doorLoc])
                    self._phase = Phase.FOLLOW_PATH_TO_DOOR

                    self.decide_if_quit(
                        len(self._navigator.get_all_waypoints()))
                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

                # Send message of current action
                self._sendMessage('Moving to door of ' +
                                  self._door['room_name'], agent_name)

            """
            Following the path to the chosen closed door

            """
            if Phase.FOLLOW_PATH_TO_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if not self.check_if_quit():
                    self._number_of_steps_to_take -= 1

                    if action != None:
                        return action, {}

                    self.decide_if_quit(0)

                    self._phase = Phase.OPEN_DOOR
                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Opening the door

            """
            if Phase.OPEN_DOOR == self._phase:
                self._sendMessage('Opening door of ' +
                                  self._door['room_name'], agent_name)

                if not self.check_if_quit():
                    enterLoc = self._door['location']
                    enterLoc = enterLoc[0], enterLoc[1] - 1

                    self._navigator.add_waypoints([enterLoc])
                    self._phase = Phase.ENTER_THE_ROOM

                    self.decide_if_quit(
                        len(self._navigator.get_all_waypoints()))

                    return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Entering the room

            """
            if Phase.ENTER_THE_ROOM == self._phase:
                self._state_tracker.update(state)
                # Enter the room
                action = self._navigator.get_move_action(self._state_tracker)

                if not self.check_if_quit():
                    self._number_of_steps_to_take -= 1

                    if action != None:
                        return action, {}

                    self._sendMessage('Entering the ' +
                                      self._door['room_name'], agent_name)

                    self._phase = Phase.SCAN_ROOM

                    room_name = self._door['room_name']
                    self.visit_new_room(room_name)

                    self.decide_if_quit(0)

                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Searching the room

            """
            if Phase.SCAN_ROOM == self._phase:
                self._navigator.reset_full()

                roomInfo = state.get_room_objects(self._door['room_name'])
                roomArea = [area['location'] for area in roomInfo if area['name']
                            == self._door['room_name'] + "_area"]

                if not self.check_if_quit():
                    self._navigator.add_waypoints(roomArea)
                    self._phase = Phase.SEARCH_AND_FIND_GOAL_BLOCK

                    self.decide_if_quit(
                        len(self._navigator.get_all_waypoints()))
                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

                self._sendMessage('Scanning ' +
                                  self._door['room_name'], agent_name)

            """
            Looking for the goal block
            if found then grab it and drop it either at :
                a) The Drop zone
                b) The Intermidiate storage

            """
            if Phase.SEARCH_AND_FIND_GOAL_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)

                if not self.check_if_quit():
                    self._number_of_steps_to_take -= 1

                    roomObjects = state.get_closest_with_property(
                        'is_goal_block')
                    roomObjects = [
                        x for x in roomObjects if x['is_collectable'] == True]

                    for obj in roomObjects:
                        result = self._checkIfDesiredBlock(obj)

                        if result[0]:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([result[1]])

                            self.decide_if_quit(0)

                            self._sendMessage(
                                'Spotted goal object ' + result[2] + ' at ' + self._door['room_name'] + ", Index: " + str(result[4]), agent_name)

                            if not self.check_if_quit():
                                if result[3]:
                                    self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT

                                    self.decide_if_quit(
                                        len(self._navigator.get_all_waypoints()))

                                else:
                                    self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE

                                    self.decide_if_quit(
                                        len(self._navigator.get_all_waypoints()))

                                if self._currentlyCarrying != -1:
                                    objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                                    visualizationObj = str(
                                        state[self.agent_id]['is_carrying'][0]['visualization'])

                                    self._sendMessage('Stored nearby the goal object ' + '{' + objCarryId + "}" + "with " +
                                                      "[" + visualizationObj + "]" + ", Index: " + str(self._currentlyCarrying), agent_name)
                                    self._phase = Phase.PLAN_PATH_TO_DOOR
                                    return DropObject.__name__, {'object_id': objCarryId}

                                self._currentlyCarrying = result[4]
                                return GrabObject.__name__, {'object_id': obj['obj_id']}

                            else:
                                self._phase = Phase.PLAN_PATH_TO_DOOR

                        room_name = self._door['room_name']
                        self.discover_block_in_visited_room(obj, room_name)

                    if action != None:
                        return action, {}

                    self._phase = Phase.PLAN_PATH_TO_DOOR
                    self.decide_if_quit(0)

                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Plan to drop currently desired object
            at the drop zone

            """
            if Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT == self._phase:
                if self._checkIfCurrentlyCarrying(state):
                    self._state_tracker.update(state)
                    # Follow path to door
                    action = self._navigator.get_move_action(
                        self._state_tracker)

                    if not self.check_if_quit():
                        self._number_of_steps_to_take -= 1

                        if action != None:
                            return action, {}

                        self.decide_if_quit(0)

                        if not self.check_if_quit():
                            self._sendMessage(
                                'Put currently desired object ' + str(self._currentlyCarrying), agent_name)

                            if self._currentlyWantedBlock < len(self._goalBlockCharacteristics) - 1:
                                self._currentlyWantedBlock += 1

                            objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                            self._currentlyCarrying = -1

                            self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                            self.decide_if_quit(0)

                            return DropObject.__name__, {'object_id': objCarryId}

                        else:
                            self._phase = Phase.PLAN_PATH_TO_DOOR

                    else:
                        self._phase = Phase.PLAN_PATH_TO_DOOR

                else:
                    self._currentlyCarrying = -1
                    self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY

            """
            Plan to drop goal object to the next to the drop zone
            a.k.a. the intermediate storage

            """
            if Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE == self._phase:
                if self._checkIfCurrentlyCarrying(state):
                    self._state_tracker.update(state)
                    # Follow path to door
                    action = self._navigator.get_move_action(
                        self._state_tracker)

                    if not self.check_if_quit():
                        self._number_of_steps_to_take -= 1

                        if action != None:
                            return action, {}

                        self.decide_if_quit(0)

                        if not self.check_if_quit():
                            objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                            visualizationObj = str(
                                state[self.agent_id]['is_carrying'][0]['visualization'])

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

                            self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                            self.decide_if_quit(0)

                            return DropObject.__name__, {'object_id': objCarryId}

                        else:
                            self._phase = Phase.PLAN_PATH_TO_DOOR

                    else:
                        self._phase = Phase.PLAN_PATH_TO_DOOR

                else:
                    self._currentlyCarrying = -1
                    self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY

            """
            Searching for the currenly desired goal block in the intermediate storage.
            If found then grab it

            """
            if Phase.GRAB_DESIRED_OBJECT_NEARBY == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if not self.check_if_quit():
                    self._number_of_steps_to_take -= 1

                    if action != None:
                        return action, {}

                    self.decide_if_quit(0)

                    if not self.check_if_quit():
                        block_location = self._goalBlockCharacteristics[
                            self._currentlyWantedBlock]['location']
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block_location])

                        for storedBlockID in self._nearbyGoalBlocksStored[self._droppedBlockIndex]:
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
                                self._nearbyGoalBlocksStored[self._droppedBlockIndex].remove(
                                    storedBlockID)
                                self._currentlyCarrying = self._currentlyWantedBlock

                                self._phase = Phase.PLAN_TO_DROP_CURRENTLY_DESIRED_OBJECT
                                self.decide_if_quit(
                                    len(self._navigator.get_all_waypoints()))

                                return GrabObject.__name__, {'object_id': storedBlockID[0]}

                    else:
                        self._phase = Phase.PLAN_PATH_TO_DOOR

                else:
                    self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Check if the currently desired goal block is in the intermediate storage.
            If not then go to the rooms.

            """
            if Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY == self._phase:
                self._navigator.reset_full()
                if self._currentlyWantedBlock >= len(self._goalBlockCharacteristics):
                    self._currentlyWantedBlock = len(
                        self._goalBlockCharacteristics) - 1

                goalBlockVisualization = self._goalBlockCharacteristics[
                    self._currentlyWantedBlock]['visualization']

                for index in self._nearbyGoalBlocksStored:
                    for droppedBlock in self._nearbyGoalBlocksStored[index]:
                        storedSize = float(droppedBlock[1][droppedBlock[1].find(
                            "'size': ")+8:droppedBlock[1].find(",")])
                        storedShape = int(droppedBlock[1][droppedBlock[1].find(
                            "'shape': ")+9:droppedBlock[1].find(", 'co")])
                        storedColour = droppedBlock[1][droppedBlock[1].find(
                            "'colour': ")+11:droppedBlock[1].find(", 'de") - 1]

                        if int(goalBlockVisualization['shape']) == storedShape and goalBlockVisualization['colour'] == storedColour and float(goalBlockVisualization['size']) == storedSize:

                            self._droppedBlockIndex = index

                            # Really a hack to find the location of the dropped block because they are defined for all 3 goal blocks, not for every dropped block
                            block_location = self._goalBlockCharacteristics[index]['location']
                            block_location = block_location[0] + \
                                3, block_location[1]
                            self._navigator.add_waypoints([block_location])

                            self._phase = Phase.GRAB_DESIRED_OBJECT_NEARBY

                            action = self._navigator.get_move_action(
                                self._state_tracker)
                            return action, {}

                self._phase = Phase.PLAN_PATH_TO_DOOR
