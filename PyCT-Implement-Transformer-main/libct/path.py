# Copyright: see copyright.txt

import logging
from libct.constraint import Constraint
from libct.predicate import Predicate
from libct.utils import unwrap

log = logging.getLogger("ct.path")

class PathToConstraint:
    root_constraint = None

    def __init__(self):
        if self.root_constraint is None:
            self.root_constraint = Constraint(None, None)
        self.current_constraint = self.root_constraint

    def add_branch(self, conbool):
        # print("add_branch")
        # print("conbool:", conbool.expr, unwrap(conbool))
        p = Predicate(conbool.expr, unwrap(conbool))
        # print("p:")
        c = self.current_constraint.find_child(p)
        # print("c:")
        pneg = Predicate(conbool.expr, not unwrap(conbool))
        # print("pneg:")
        cneg = self.current_constraint.find_child(pneg)
        # print("cneg:")
        if c is None and cneg is None:
            c = self.current_constraint.add_child(p); c.processed = True # for debugging purposes
            cneg = self.current_constraint.add_child(pneg)
            conbool.engine.constraints_to_solve.append(cneg) # add the negated constraint to the queue for later traversal
            # print("conbool.engine.constraints_to_solve:")
            # log.smtlib2(f"Now constraint: {c}"); log.smtlib2(f"Add constraint: {cneg}")
        else:
            assert c is not None and cneg is not None
            # print("line33")
        # print('change current_constraint to c')
        self.current_constraint = c # move the current constraint to the child we want now
        # print('end add_branch')
