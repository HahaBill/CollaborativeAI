from cmath import nan
import logging
import scipy as sc
import numpy as np

from random import randint
from typing import Tuple, cast

from sklearn.preprocessing import LabelEncoder
from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from sqlalchemy import false, true
from scipy.stats import mode, variation


class StalinAgent(DefaultParty):
    """
    Template agent that offers random bids until a bid with sufficient utility is offered.
    """

    def __init__(self):
        super().__init__()
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile = None
        self._last_received_bid: Bid = None
        self.best_offer_opponent: Bid = None
        self.best_bid: Bid = None
        self.calculated_bid: bool = False
        self.opponent_issues = {}
        self.opp_history_bids = ({})
        self.sorted_bid = ()

        self.issues = []
        self.bid_history = {}
        self.opp_profile = {}

        # Issues to numeric
        self.issue_to_numeric = {}
        self.idx_issue = 1

        # Values to numeric
        self.value_to_numeric = {}
        self.idx_value = 1

    def notifyChange(self, info: Inform):
        """This is the entry point of all interaction with your agent after is has been initialised.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(info, Settings):
            self._settings: Settings = cast(Settings, info)
            self._me = self._settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self._progress = self._settings.getProgress()

            # the profile contains the preferences of the agent over the domain
            self._profile = ProfileConnectionFactory.create(
                info.getProfile().getURI(), self.getReporter()
            )
        # ActionDone is an action send by an opponent (an offer or an accept)
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()

            # if it is an offer, set the last received bid
            if isinstance(action, Offer):
                self._last_received_bid = cast(Offer, action).getBid()
        # YourTurn notifies you that it is your turn to act
        elif isinstance(info, YourTurn):
            # execute a turn
            self._myTurn()

            # log that we advanced a turn
            self._progress = self._progress.advance()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(info, Finished):
            # terminate the agent MUST BE CALLED
            self.terminate()
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # lets the geniusweb system know what settings this agent can handle
    # leave it as it is for this course
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    # terminates the agent and its connections
    # leave it as it is for this course
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profile is not None:
            self._profile.close()
            self._profile = None

    #######################################################################################
    ########## THE METHODS BELOW THIS COMMENT ARE OF MAIN INTEREST TO THE COURSE ##########
    #######################################################################################

    # give a description of your agent
    def getDescription(self) -> str:
        return "Agent Stalin for Collaborative AI course"

    # execute a turn
    def _myTurn(self):

        print("\n \n ")
        print(self.opponent_issues)
        print("\n \n ")

        profile = self._profile.getProfile()
        progress = self._progress.get(0)

        if(self.best_offer_opponent is None):
            self.best_offer_opponent = self._last_received_bid

        if(profile.getUtility(self._last_received_bid) > profile.getUtility(self.best_offer_opponent)):
            self.best_offer_opponent = self._last_received_bid

        if(progress > 0.7):
            action = Accept(self._me, self.best_offer_opponent)

        # Update the opponent profile with the new bid
        self.update_bid_history(self._last_received_bid)
        self.analyse_opp_profile()
        self.sort_high_bids()

        # check if the last received offer if the opponent is good enough
        if self._isGood(self._last_received_bid):
            # if so, accept the offer
            action = Accept(self._me, self._last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self._findBid()
            action = Offer(self._me, bid)

        # send the action
        #  self.update_opponent_issues()
        self.getConnection().send(action)

    def _isGood(self, bid: Bid) -> bool:
        if bid is None:
            return False
        profile = self._profile.getProfile()
        progress = self._progress.get(0)

        # very basic approach that accepts if the offer is valued above 0.6 and
        # 80% of the rounds towards the deadline have passed
        # AC_NEXT
        return profile.getUtility(bid) > profile.getUtility(self._findBid()) and progress > 0.8

    def _findBid(self) -> Bid:
        """
        Finds the best offer to maximize the win.
        """
        return self.find_best_offer()

    # Opponent Modelling

    def update_bid_history(self, bid):
        # Add new bid/offer to the history
        """
        Adding new bid/offer to the history, if issue is not recorded in the
        "issue_to_numeric". Assign a numeric representation of the issue and save it in the
        dict.

        Same for values, if there are missing numerical representation for them. Create one in
        "value_to_numeric" dict.
        """
        bid_dict = bid.getIssueValues().items()
        for issue, value in bid_dict:
            if issue not in self.issue_to_numeric:
                self.map_issues_to_numeric_and_initialize(issue)

            if value not in self.value_to_numeric:
                self.map_value_to_numeric(value)

            idx_numeric_issue = self.issue_to_numeric[issue]
            self.bid_history[idx_numeric_issue].append(value)

    def map_issues_to_numeric_and_initialize(self, issue):
        """
        Map issues which are represented in String to numeric values,
        furthermore it initializes issue-history pair in "bid_history" dict.
        """
        self.issue_to_numeric[issue] = self.idx_issue
        self.bid_history[self.idx_issue] = []
        self.idx_issue = self.idx_issue + 1

    def map_value_to_numeric(self, value):
        """
        Map values which are represented in String to numeric values.
        """
        self.value_to_numeric[value] = self.idx_value
        self.idx_value = self.idx_value + 1

    def analyse_opp_profile(self):
        # Calculate the mode and the variance of values per issue
        for issue, values in self.bid_history.items():
            if issue not in self.opp_profile:
                self.opp_profile[issue] = ()

            numerical_values = [self.value_to_numeric[value]
                                for value in values]

            if len(numerical_values) == 1:
                self.opp_profile[issue] = (
                    0, 0)
            else:
                self.opp_profile[issue] = (
                    mode(numerical_values), np.var([1, 2, 3, 4], ddof=1))
    # Bidding

    def find_best_offer(self) -> Bid:
        # Find the best possible offer given opponent profile
        for bid in self.sorted_bid:
            found = True
            for issue, value in bid.getIssueValues().items():
                if self.accept(issue, value):
                    continue
                else:
                    found = False
                    break
            if(found):
                print("FOUND BID : ", bid)
                return bid

        return self.sorted_bid[len(self.sorted_bid) - 1]

    def accept(self, issue, value):
        # Check if the give nvalue is within the acceptable range
        issue_range = self.get_acceptable_range(issue)
        if value not in self.value_to_numeric:
            self.map_value_to_numeric(value)

        low = issue_range[0]
        high = issue_range[1]
        if type(issue_range[0]) is np.ndarray:
            low = issue_range[0][0][0]
            high = issue_range[1][0][0]
            print(low)
            print("VARIANCE : ", issue_range[1])
        return low <= self.value_to_numeric[value] <= high

    def get_acceptable_range(self, issue):
        numerical_issue = self.issue_to_numeric[issue]
        issue_mode = self.opp_profile[numerical_issue][0]
        issue_var = self.opp_profile[numerical_issue][1]

        low = issue_mode - issue_var
        high = issue_mode + issue_var

        return low, high

    ####
    def sort_high_bids(self):
        """
        Sorting bids based on the utility values
        """
        temp_tuple_bid = []
        if(not self.calculated_bid):
            domain = self._profile.getProfile().getDomain()
            all_bids = AllBidsList(domain)
            profile = self._profile.getProfile()

            for x in all_bids:
                temp_tuple_bid.append((profile.getUtility(x), x))

            temp_tuple_bid = sorted(temp_tuple_bid, key=lambda x: x[0])

            self.calculated_bid = True
            self.sorted_bid = [bid[1] for bid in temp_tuple_bid]

    def update_opponent_issues(self):
        """
        Keep track of frequencies of the values in bids received by
        the opponents over period of time.
        """

        recentIssues = self._last_received_bid.getIssues()
        recentIssuesValues = self._last_received_bid.getIssueValues()

        for issue in recentIssues:
            if issue in self.opponent_issues:
                if recentIssuesValues[issue] in self.opponent_issues[issue]:
                    self.opponent_issues[issue] = self.opponent_issues[issue][recentIssuesValues[issue]] + 1
                else:
                    self.opponent_issues[issue][recentIssuesValues[issue]] = 1
            else:
                self.opponent_issues[issue][recentIssuesValues[issue]] = 1

    def update_history_opp_issues(self):
        recentIssuesValues = self._last_received_bid.getIssueValues()
        self.opp_history_bids.append(recentIssuesValues)

    def always_best_bid_init(self) -> Bid:
        """
        Returns the best bid
        """
        if(not self.calculated_bid):
            domain = self._profile.getProfile().getDomain()
            all_bids = AllBidsList(domain)
            profile = self._profile.getProfile()

            best_utility = 0.0

            for x in all_bids:
                curr_utility = profile.getUtility(x)
                if(best_utility < curr_utility):
                    bid = x
                    best_utility = curr_utility

            self.calculated_bid = True
            self.best_bid = bid

            return self.best_bid
        else:
            return self.best_bid
