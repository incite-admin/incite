# Copyright: see copyright.txt
from __future__ import annotations
from libct.predicate import Predicate


class Constraint:
    global_constraints: list[Self] = []

    def __init__(self, parent: int | None, last_predicate, height: int = 0):
        self.parent: int | None = parent # should be "None" or "a Constraint id"
        self.last_predicate = last_predicate # should be "None" or "a Predicate"
        self.children: list[int] = [] # a list of "Constraint id"s
        self.height: int = height # for debugging purposes
        self.processed: bool = False # for debugging purposes
        self.id: int = len(self.global_constraints)
        self.global_constraints.append(self)

    def __eq__(self, other: Self):
        """Two Constraints are equal iff they have the same chain of predicates"""
        return isinstance(other, self.__class__) and \
            self.parent == other.parent and \
            self.last_predicate == other.last_predicate

    def __str__(self):
        return str(list(map(lambda x: str(x), self.get_all_asserts()))) + f"  (path_len: {self.height})"
        # return str(self.last_predicate) + f"  (processed: {self.processed}, path_len: {self.height})"

    def add_child(self, predicate):
        # assert self.find_child(predicate) is None
        c = Constraint(self.id, predicate, self.height + 1)
        self.children.append(c.id)
        return c

    def find_child(self, predicate) -> Self:
        return next((self.global_constraints[c] for c in self.children if self.global_constraints[c].last_predicate == predicate), None)

    def get_all_asserts(self):
        self.processed = True # for debugging purposes
        asserts = []; tmp = self
        while tmp.last_predicate is not None: # collect all the assertions in this constraint path
            asserts.append(tmp.last_predicate)
            tmp = self.global_constraints[tmp.parent]
        return asserts[::-1]
