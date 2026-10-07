"""Timeline references preserve exact output order without duplicating data."""
import pytest

from openecon.output_events import validate_output_events


def test_truncated_unicode_buffer_keeps_a_stable_prefix_after_display_boundary():
    from openecon.console_worker import BoundedText, MAX_STDOUT
    output = BoundedText()
    output.write('a' * (MAX_STDOUT - 1))
    output.write('ğ')
    before_display = output.getvalue()
    output.write('x')
    assert output.getvalue() == before_display
    assert before_display.count('[Output truncated') == 1


def test_unicode_stdout_partition_and_each_output_are_preserved():
    validate_output_events([
        {'type': 'stdout', 'text': 'önce\n'},
        {'type': 'output', 'index': 0},
        {'type': 'stdout', 'text': 'sonra\n'},
    ], 'önce\nsonra\n', [{'type': 'text', 'data': 'value'}])
    validate_output_events([], '', [])
    validate_output_events([{'type': 'stdout', 'text': 'hello'}], 'hello', [])


@pytest.mark.parametrize('events,stdout,outputs', [
    (None, '', []),
    ([{'type': 'stdout', 'text': ''}], '', []),
    ([{'type': 'stdout', 'text': 'a'}, {'type': 'stdout', 'text': 'b'}], 'ab', []),
    ([{'type': 'stdout', 'text': 'wrong'}], 'correct', []),
    ([], '', [{}]),
    ([{'type': 'output', 'index': True}], '', [{}]),
    ([{'type': 'output', 'index': -1}], '', [{}]),
    ([{'type': 'output', 'index': 1}, {'type': 'output', 'index': 0}], '', [{}, {}]),
    ([{'type': 'output', 'index': 0}, {'type': 'output', 'index': 0}], '', [{}, {}]),
    ([{'type': 'output', 'index': 0, 'html': '<script>'}], '', [{}]),
    ([{'type': 'arbitrary', 'text': 'a'}], 'a', []),
    ([{'type': 'stdout', 'text': 'ğ' * 36000}], 'ğ' * 36000, []),
])
def test_malformed_or_incomplete_timeline_is_rejected(events, stdout, outputs):
    with pytest.raises(ValueError):
        validate_output_events(events, stdout, outputs)
