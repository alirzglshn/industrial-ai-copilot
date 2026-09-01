"""safe_eval: arithmetic without eval().

The model controls the expression string, so this is tested as a security
boundary, not just a calculator: everything that is not one of +, -, *, /,
//, %, **, unary +/-, and numeric literals must be rejected, whatever form it
takes.
"""

import pytest

from copilot.agent.tools import CalculatorError, CalculatorTool, safe_eval


def test_basic_arithmetic() -> None:
    assert safe_eval("2 + 3") == 5.0
    assert safe_eval("10 - 4") == 6.0
    assert safe_eval("6 * 7") == 42.0
    assert safe_eval("10 / 4") == 2.5


def test_percentage_difference() -> None:
    """The exact example from the project spec: comparing two specifications."""
    assert safe_eval("(95-80)/80*100") == pytest.approx(18.75)


def test_operator_precedence_and_parentheses() -> None:
    assert safe_eval("2 + 3 * 4") == 14.0
    assert safe_eval("(2 + 3) * 4") == 20.0


def test_unary_minus_and_plus() -> None:
    assert safe_eval("-5 + 3") == -2.0
    assert safe_eval("+5 - 3") == 2.0
    assert safe_eval("-(2 + 3)") == -5.0


def test_floor_division_and_modulo() -> None:
    assert safe_eval("7 // 2") == 3.0
    assert safe_eval("7 % 2") == 1.0


def test_exponentiation() -> None:
    assert safe_eval("2 ** 10") == 1024.0


def test_float_literals() -> None:
    assert safe_eval("1.5 + 2.5") == 4.0


def test_nested_parentheses() -> None:
    assert safe_eval("((1 + 2) * (3 + 4))") == 21.0


def test_division_by_zero_is_rejected() -> None:
    with pytest.raises(CalculatorError, match="[Dd]ivision"):
        safe_eval("1 / 0")


def test_malformed_expression_is_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("2 + * 3")


def test_empty_expression_is_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("")


def test_names_are_rejected() -> None:
    """No variable lookup: nothing in this tool's world has a name to resolve."""
    with pytest.raises(CalculatorError):
        safe_eval("x + 1")


def test_function_calls_are_rejected() -> None:
    """This is the injection-shaped case: no path from an expression to code execution."""
    with pytest.raises(CalculatorError):
        safe_eval("__import__('os').system('echo hi')")


def test_attribute_access_is_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("(1).__class__")


def test_list_and_comprehension_syntax_is_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("[x for x in range(10)]")


def test_string_literals_are_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("'a' + 'b'")


def test_boolean_literals_are_rejected() -> None:
    """bool is a subclass of int in Python; explicitly excluded so True != 1 here."""
    with pytest.raises(CalculatorError):
        safe_eval("True + 1")


def test_comparison_operators_are_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("1 < 2")


def test_huge_exponent_is_rejected() -> None:
    """A bound against a crafted expression burning CPU on a huge power."""
    with pytest.raises(CalculatorError, match="[Ee]xponent"):
        safe_eval("9 ** 99999999999")


def test_negative_exponent_within_bound_is_allowed() -> None:
    assert safe_eval("2 ** -2") == 0.25


def test_multiline_or_statement_syntax_is_rejected() -> None:
    with pytest.raises(CalculatorError):
        safe_eval("import os")


class TestCalculatorTool:
    def test_run_returns_the_numeric_result(self) -> None:
        result = CalculatorTool().run(expression="(95-80)/80*100")

        assert result.tool_name == "calculate"
        assert result.output == pytest.approx(18.75)

    def test_run_propagates_calculator_errors(self) -> None:
        with pytest.raises(CalculatorError):
            CalculatorTool().run(expression="1/0")
