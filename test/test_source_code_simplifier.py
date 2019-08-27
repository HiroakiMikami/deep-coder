import unittest

from src.dsl import Function, Type, Variable, Expression, Program, Signature, Statement
from src.source_code_simplifier import normalize, remove_redundant_variables, remove_redundant_expressions, remove_dependency_between_variables


class Test_source_code_simplifier(unittest.TestCase):
    def test_normalize(self):
        F = Function("FUNC", Signature([Type.Int], Type.IntList))
        p = Program([Variable(1, Type.Int), Variable(0, Type.IntList)], [])
        p = normalize(p)
        self.assertEqual(
            p,
            Program([Variable(0, Type.IntList), Variable(1, Type.Int)], [])
        )

        p = Program([Variable(2, Type.Int), Variable(0, Type.IntList)], [
                    Statement(Variable(3, Type.Int), Expression(F, [Variable(2, Type.Int)]))])
        p = normalize(p)
        self.assertEqual(
            p,
            Program([Variable(0, Type.IntList), Variable(1, Type.Int)], [
                    Statement(Variable(2, Type.Int), Expression(F, [Variable(1, Type.Int)]))])
        )

    def test_remove_redundant_variables(self):
        F = Function("FUNC", Signature([Type.Int], Type.IntList))
        p = Program([Variable(0, Type.Int), Variable(1, Type.IntList)], [])
        p = remove_redundant_variables(p)
        self.assertEqual(p, Program([], []))

        p = Program([Variable(0, Type.Int), Variable(1, Type.IntList)], [
                    Statement(Variable(2, Type.Int), Expression(F, [Variable(0, Type.Int)]))])
        p = remove_redundant_variables(p)
        self.assertEqual(
            p,
            Program([Variable(0, Type.Int)], [
                    Statement(Variable(2, Type.Int), Expression(F, [Variable(0, Type.Int)]))])
        )

        p = Program([Variable(0, Type.Int)], [Statement(Variable(1, Type.Int), Expression(F, [Variable(
            0, Type.Int)])), Statement(Variable(2, Type.Int), Expression(F, [Variable(0, Type.Int)]))])
        p = remove_redundant_variables(p)
        self.assertEqual(
            p,
            Program([Variable(0, Type.Int)], [
                    Statement(Variable(2, Type.Int), Expression(F, [Variable(0, Type.Int)]))])
        )

    def test_remove_redundant_expressions(self):
        F = Function("F", Signature([Type.IntList], Type.IntList))
        SORT = Function("SORT", Signature([Type.IntList], Type.IntList))
        REVERSE = Function("REVERSE", Signature([Type.IntList], Type.IntList))

        p = Program([Variable(0, Type.IntList)], [
                    Statement(Variable(1, Type.Int), Expression(F, [Variable(0, Type.IntList)]))])
        p = remove_redundant_expressions(p)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
                         Statement(Variable(1, Type.Int), Expression(F, [Variable(0, Type.IntList)]))]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(2, Type.IntList)]))
        ])
        p = remove_redundant_expressions(p)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(1, Type.IntList)])),
        ]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                SORT, [Variable(1, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(2, Type.IntList)]))
        ])
        p = remove_redundant_expressions(p)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(1, Type.IntList)])),
        ]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                REVERSE, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                REVERSE, [Variable(1, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(2, Type.IntList)]))
        ])
        p = remove_redundant_expressions(p)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                REVERSE, [Variable(0, Type.IntList)])),
            Statement(Variable(3, Type.IntList), Expression(
                F, [Variable(0, Type.IntList)])),
        ]))

    def test_remove_dependency_between_variables(self):
        F = Function("F", Signature([Type.IntList], Type.IntList))
        SORT = Function("SORT", Signature([Type.IntList], Type.IntList))
        REVERSE = Function("REVERSE", Signature([Type.IntList], Type.IntList))
        MAXIMUM = Function("MAXIMUM", Signature([Type.IntList], Type.Int))
        MINIMUM = Function("MINIMUM", Signature([Type.IntList], Type.Int))
        SUM = Function("SUM", Signature([Type.IntList], Type.Int))
        HEAD = Function("HEAD", Signature([Type.IntList], Type.Int))
        LAST = Function("LAST", Signature([Type.IntList], Type.Int))

        p = Program([Variable(0, Type.IntList)], [
                    Statement(Variable(1, Type.Int), Expression(F, [Variable(0, Type.IntList)]))])
        p = remove_dependency_between_variables(p, MINIMUM, MAXIMUM)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
                         Statement(Variable(1, Type.Int), Expression(F, [Variable(0, Type.IntList)]))]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                REVERSE, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                SUM, [Variable(1, Type.IntList)]))
        ])
        p = remove_dependency_between_variables(p, MINIMUM, MAXIMUM)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                REVERSE, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                SUM, [Variable(0, Type.IntList)])),
        ]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                HEAD, [Variable(1, Type.IntList)]))
        ])
        p = remove_dependency_between_variables(p, MINIMUM, MAXIMUM)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                MINIMUM, [Variable(0, Type.IntList)])),
        ]))

        p = Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                LAST, [Variable(1, Type.IntList)]))
        ])
        p = remove_dependency_between_variables(p, MINIMUM, MAXIMUM)
        self.assertEqual(p, Program([Variable(0, Type.IntList)], [
            Statement(Variable(1, Type.IntList), Expression(
                SORT, [Variable(0, Type.IntList)])),
            Statement(Variable(2, Type.IntList), Expression(
                MAXIMUM, [Variable(0, Type.IntList)])),
        ]))


if __name__ == "__main__":
    unittest.main()
