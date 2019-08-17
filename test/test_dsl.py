import unittest

from src.dsl import Function, Type, Variable, Expression, Program, to_string

class Test_to_string(unittest.TestCase):
    def test_to_string(self):
        self.assertEqual("a <- int\nb <- [int]\n", to_string(Program([Variable(0, Type.Int), Variable(1, Type.IntList)], [])))
        F = Function("FUNC", [int, [int]], None, None)
        self.assertEqual("a <- int\nb <- [int]\nc <- FUNC b a\n", to_string(Program(
            [Variable(0, Type.Int), Variable(1, Type.IntList)],
            [(Variable(2, Type.Int), Expression(F, [Variable(1, Type.Int), Variable(0, Type.Int)]))]
        )))

if __name__ == "__main__":
    unittest.main()
