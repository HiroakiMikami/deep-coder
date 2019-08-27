import copy
from .dsl import Function, Type, Variable, Expression, Program, Statement


def normalize(program: Program):
    """
    Return the normalized program
    This function applies 2 transformations:
    1) sort input variables by order of use
    2) make the variable ids consecutive numbers

    Parameters
    ----------
    program : Program
        The program that will be normalized

    Returns
    -------
    program : Program
        The normalized program

    Notes
    -----
    This function modify the program object of the argument
    to reduce runtime overhead.
    """
    program = program.clone()  # Clone program to isolate the argument from modifications

    # inputs should be sorted by id
    program.inputs.sort(key=lambda i: i.id)

    # Re-assign id
    old_id_to_new_id = dict()
    for i in program.inputs:
        old_id = i.id
        i.id = len(old_id_to_new_id)
        old_id_to_new_id[old_id] = i.id
    for statement in program.body:
        old_id = statement.variable.id
        statement.variable.id = len(old_id_to_new_id)
        old_id_to_new_id[old_id] = statement.variable.id
        for arg in statement.expression.arguments:
            arg.id = old_id_to_new_id[arg.id]

    return program


def remove_redundant_variables(program: Program) -> Program:
    """
    Return the program that is removed the redundant variables
    For examples, the program A will be converted to the program B
    ```
    <Program A>
    a <- [int]
    b <- int
    c <- TAKE b a
    d <- REVERSE a

    <Program B>
    a <- [int]
    d <- REVERSE a
    ```

    Parameters
    ----------
    program : Program
        The program that will be simplified

    Returns
    -------
    Program
        The simplified program
    """

    program = program.clone()  # Clone program to isolate the argument from modifications

    inputs = []
    body = []
    if len(program.body) == 0:
        program.inputs.clear()
        program.body.clear()
        return program

    # Last line is always used (because it is output value)
    v_used = set([program.body[-1].variable])
    for statement in program.body[::-1]:
        if statement.variable in v_used:
            # v is not a redundant variable
            body.append(statement)
            for a in statement.expression.arguments:
                v_used.add(a)
    body.reverse()

    for v in program.inputs:
        if v in v_used:
            # v is not a redundant variable
            inputs.append(v)

    program.inputs = inputs
    program.body = body
    return program


def remove_redundant_expressions(program: Program) -> Program:
    """
    Return the program that is removed the redundant expressions
    This function applies following 3 rules:

    Rule1: The duplicated expressions will be merged
    ```
    <Program>
    a <- [int]
    b <- REVERSE a
    c <- REVERSE a
    d <- ZIPWITH * b c

    <Program returned by this function>
    a <- [int]
    b <- REVERSE a
    d <- ZIPWITH * b b
    ```

    Rule2: SORT function for the sorted list will be removed
    ```
    <Program>
    a <- [int]
    b <- SORT a
    c <- SORT a
    d <- ZIPWITH * b c

    <Program returned by this function>
    a <- [int]
    b <- SORT a
    d <- ZIPWITH * b b
    ```

    Rule3: REVERSE function for the reversed list will be removed
    ```
    <Program>
    a <- [int]
    b <- REVERSE a
    c <- REVERSE a
    d <- ZIPWITH * b c

    <Program returned by this function>
    a <- [int]
    b <- REVERSE a
    d <- ZIPWITH * b a
    ```

    Parameters
    ----------
    program : Program
        The program that will be simplified

    Returns
    -------
    Program
        The simplified program
    """
    program = program.clone()  # Clone program to isolate the argument from modifications

    replacement = dict()  # Variable -> Variable
    expression_to_variable = dict()  # (str, [Variable]) -> Variable
    variable_to_expression = dict()  # Variable -> Expression

    body = []
    for statement in program.body:
        if (statement.expression.function.name, tuple(statement.expression.arguments)) in expression_to_variable:
            # Rule1
            replacement[statement.variable] = expression_to_variable[(
                statement.expression.function.name, tuple(statement.expression.arguments))]
            continue
        if len(statement.expression.arguments) > 0 and statement.expression.arguments[0] in variable_to_expression:
            exp_arg1 = variable_to_expression[statement.expression.arguments[0]]
            if statement.expression.function.name == "SORT" and (exp_arg1.function.name == "SORT"):
                # Rule2
                replacement[statement.variable] = statement.expression.arguments[0]
                continue
            if statement.expression.function.name == "REVERSE" and (exp_arg1.function.name == "REVERSE"):
                # Rule3
                replacement[statement.variable] = variable_to_expression[statement.expression.arguments[0]].arguments[0]
                continue

        for i, arg in enumerate(statement.expression.arguments):
            if arg in replacement:
                statement.expression.arguments[i] = replacement[arg]
        body.append(statement)

        expression_to_variable[(statement.expression.function.name, tuple(
            statement.expression.arguments))] = statement.variable
        variable_to_expression[statement.variable] = statement.expression
    program.body = body

    return program


def remove_dependency_between_variables(program: Program, minimum: Function, maximum: Function) -> Program:
    """
    Return the program that is reduced dependencies between variables
    This function applies following 3 rules:

    Rule1: Reordering functions (REVERSE, SORT) before reduce functions
           (SUM, MAXIMUM, MINIMUM) will be ignored
    ```
    <Program>
    a <- [int]
    b <- REVERSE a
    c <- SUM b

    <Program returned by this function>
    a <- [int]
    b <- REVERSE a
    c <- SUM a
    ```

    Rule2: HEAD function to a sorted list will be converted
           into MINIMUM function
    ```
    <Program>
    a <- [int]
    b <- SORT a
    c <- HEAD b

    <Program returned by this function>
    a <- [int]
    b <- SORT a
    c <- MINIMUM a
    ```

    Rule3: LAST function to a sorted list will be converted
           into MAXIMUM function
    ```
    <Program>
    a <- [int]
    b <- SORT a
    c <- LAST b

    <Program returned by this function>
    a <- [int]
    b <- SORT a
    c <- MAXIMUM a
    ```

    Parameters
    ----------
    program : Program
        The program that will be simplified
    minimum : Function
        The MINIMUM function
    maximum : Function
        The MAXIMUM function

    Returns
    -------
    Program
        The simplified program
    """
    program = program.clone()  # Clone program to isolate the argument from modifications

    variable_to_expression = dict()  # Variable -> Expression

    body = []
    for statement in program.body:
        if len(statement.expression.arguments) > 0 and statement.expression.arguments[0] in variable_to_expression:
            exp_arg1 = variable_to_expression[statement.expression.arguments[0]]
            if (exp_arg1.function.name == "SORT" or exp_arg1.function.name == "REVERSE") and (statement.expression.function.name == "SUM" or statement.expression.function.name == "MAXIMUM" or statement.expression.function.name == "MINIMUM"):
                # Rule1
                x = exp_arg1.arguments[0]
                exp = Expression(statement.expression.function, [x])
                body.append(Statement(statement.variable, exp))
                variable_to_expression[statement.variable] = statement.expression
                continue
            if exp_arg1.function.name == "SORT" and (statement.expression.function.name == "HEAD"):
                # Rule2
                x = exp_arg1.arguments[0]
                exp = Expression(minimum, [x])
                body.append(Statement(statement.variable, exp))
                variable_to_expression[statement.variable] = exp
                continue
            if exp_arg1.function.name == "SORT" and (statement.expression.function.name == "LAST"):
                # Rule3
                x = exp_arg1.arguments[0]
                exp = Expression(maximum, [x])
                body.append(Statement(statement.variable, exp))
                variable_to_expression[statement.variable] = exp
                continue

        body.append(statement)
        variable_to_expression[statement.variable] = statement.expression

    program.body = body
    return program
