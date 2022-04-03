
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
The BlindAgent


Additional features from the BaselineAgent :


"""


class BlindAgent(BaseLineAgent):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._currentlyDesiredShape = ''
        self._currentlyDesiredSize = ''

    def _checkIfDesiredBlock(self, curr_obj):
        """
        The function takes the nearby object that is visible to the agent
        and assessed it whether it is a goal object or not.
        """
        c_size = curr_obj['visualization']['size']
        c_shape = curr_obj['visualization']['shape']

        if (c_size == self._currentlyDesiredSize and c_shape == self._currentlyDesiredShape):
            obj_description = "{size : " + str(c_size) + ", " + \
                "shape : " + str(c_shape) + ", " + \
                "colour : " + " " + ", " + \
                "order : " + str(index) + "}"

            dropzoneLocation = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']
            return (True, (dropzoneLocation[0] + 3, dropzoneLocation[1]), obj_description, False, self._currentlyWantedBlock)

        return (False, None)

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
                        else:
                            if objCarryId not in self._nearbyGoalBlocksStored[index_obj]:
                                self._nearbyGoalBlocksStored[index_obj].append(
                                    (objCarryId, visualizationObj))

            # Updating the current wanted goal object to get its shape and size
            if(self._currentlyWantedBlock < len(self._goalBlockCharacteristics) - 1):
                self._currentlyDesiredShape = self._goalBlockCharacteristics[
                    self._currentlyWantedBlock]['visualization']['shape']
                self._currentlyDesiredSize = self._goalBlockCharacteristics[
                    self._currentlyWantedBlock]['visualization']['size']

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

                # Send message of current action
                self._message_moving_to_door(
                    self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_DOOR

            """
            Following the path to the chosen closed door

            """
            if Phase.FOLLOW_PATH_TO_DOOR == self._phase:
                self._state_tracker.update(state)
                # Follow path to door
                action = self._navigator.get_move_action(self._state_tracker)

                if action != None:
                    return action, {}
                self._phase = Phase.OPEN_DOOR

            """
            Opening the door

            """
            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.ENTER_THE_ROOM
                # Open door
                self._message_opening_door(self._door['room_name'], agent_name)

                enterLoc = self._door['location']
                enterLoc = enterLoc[0], enterLoc[1] - 1
                self._navigator.add_waypoints([enterLoc])

                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            """
            Entering the room

            """
            if Phase.ENTER_THE_ROOM == self._phase:
                self._state_tracker.update(state)

                # Enter the room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}

                self._message_entering_room(
                    self._door['room_name'], agent_name)
                self._phase = Phase.SCAN_ROOM

                room_name = self._door['room_name']
                self.visit_new_room(room_name)

            """
            Searching the room

            """
            if Phase.SCAN_ROOM == self._phase:
                self._navigator.reset_full()
                roomInfo = state.get_room_objects(self._door['room_name'])
                roomArea = [area['location'] for area in roomInfo if area['name']
                            == self._door['room_name'] + "_area"]
                self._navigator.add_waypoints(roomArea)

                self._message_searching_room(
                    self._door['room_name'], agent_name)

                self._phase = Phase.SEARCH_AND_FIND_GOAL_BLOCK

            """
            Looking for the goal block
            if found then grab it and drop it either at :
                a) The Drop zone
                b) The Intermidiate storage

            """
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

                        block_vis = result[2]
                        block_location = result[1]
                        self._message_found_block(
                            block_vis, block_location, agent_name)

                        self._phase = Phase.PLAN_TO_DROP_GOAL_OBJECT_NEXT_TO_DROP_ZONE
                        self._message_leaving_room(
                            room_name=self._door['room_name'], sender=agent_name)

                        self._message_picking_up_block(
                            block_vis=block_vis, block_location=block_location, sender=agent_name)
                        return GrabObject.__name__, {'object_id': obj['obj_id']}

                    room_name = self._door['room_name']
                    self.discover_block_in_visited_room(obj, room_name)

                if action != None:
                    return action, {}

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

                    if action != None:
                        return action, {}

                    block_vis = state[self.agent_id]['is_carrying'][0]['visualization']
                    block_location = self._goalBlockCharacteristics[self._currentlyCarrying]['location']
                    self._message_put_currently_desired(
                        self._currentlyCarrying, agent_name)
                    self._message_droping_block(
                        block_vis, block_location, agent_name)

                    if self._currentlyWantedBlock < len(self._goalBlockCharacteristics) - 1:
                        self._currentlyWantedBlock += 1

                    objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']
                    self._phase = Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY
                    self._currentlyCarrying = -1

                    return DropObject.__name__, {'object_id': objCarryId}
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

                    if action != None:
                        return action, {}

                    objCarryId = state[self.agent_id]['is_carrying'][0]['obj_id']

                    visualizationObj = str(
                        state[self.agent_id]['is_carrying'][0]['visualization'])
                    self._phase = Phase.PLAN_PATH_TO_DOOR
                    self._message_stored_nearby(block_id=objCarryId, block_vis=visualizationObj, index=str(
                        self._currentlyCarrying), sender=agent_name)

                    self._checkGoalBlocksPlacedNearby[self._currentlyCarrying] = True
                    if self._currentlyCarrying not in self._nearbyGoalBlocksStored:
                        list_obj = []
                        list_obj.append((objCarryId, visualizationObj))
                        self._nearbyGoalBlocksStored[self._currentlyCarrying] = list_obj
                    else:
                        self._nearbyGoalBlocksStored[self._currentlyCarrying].append(
                            (objCarryId, visualizationObj))

                    self._currentlyCarrying = -1

                    return DropObject.__name__, {'object_id': objCarryId}

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

                if action != None:
                    return action, {}

                block_location = self._goalBlockCharacteristics[self._currentlyWantedBlock]['location']
                self._navigator.reset_full()
                self._navigator.add_waypoints([block_location])

                for storedBlockID in self._nearbyGoalBlocksStored[self._droppedBlockIndex]:
                    desiredBlock = self._goalBlockCharacteristics[
                        self._currentlyWantedBlock]['visualization']

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

                        self._message_picking_up_block(block_vis=desiredBlock, block_location=self._goalBlockCharacteristics[
                            self._currentlyWantedBlock]['location'], sender=agent_name)
                        return GrabObject.__name__, {'object_id': storedBlockID[0]}

                self._phase = Phase.PLAN_PATH_TO_DOOR

            """
            Check if the currently desired goal block is in the intermediate storage.
            If not then go to the rooms.

            """
            if Phase.CHECK_IF_ANOTHER_GOAL_BLOCK_PLACED_NEARBY == self._phase:
                self._navigator.reset_full()

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
