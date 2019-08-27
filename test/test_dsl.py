import unittest

from src.dsl import Function, Type, Variable, Expression, Program, Signature, Statement


class Test_Program(unittest.TestCase):
    def test_to_string(self):
        self.assertEqual("a <- int\nb <- [int]\n",
                         Program([Variable(0, Type.Int), Variable(1, Type.IntList)], []).to_string())
        F = Function("FUNC", Signature([Type.Int], Type.IntList))
        self.assertEqual("a <- int\nb <- [int]\nc <- FUNC b a\n", Program(
            [Variable(0, Type.Int), Variable(1, Type.IntList)],
            [Statement(Variable(2, Type.Int), Expression(
                F, [Variable(1, Type.Int), Variable(0, Type.Int)]))]
        ).to_string())

    def test_clone(self):
        F = Function("FUNC", Signature([Type.Int, Type.IntList], Type.IntList))
        a = Variable(0, Type.Int)
        b = Variable(1, Type.IntList)
        c = Variable(1, Type.IntList)
        p = Program([a, b], [Statement(c, Expression(F, [a, b]))])
        p_clone = p.clone()
        self.assertEqual(p, p_clone)

        p_clone.inputs[0].id = 2
        self.assertEqual(0, p.inputs[0].id)
        self.assertEqual(0, p_clone.body[0].expression.arguments[0].id)


if __name__ == "__main__":
    unittest.main()
