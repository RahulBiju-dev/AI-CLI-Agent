import json
import unittest
from unittest.mock import patch, MagicMock
from agent.cancellation import CancellationToken, OperationCancelled
from tools.automated_routine_executor import automated_routine_executor, _validate, _run_action

class TestAutomatedRoutineExecutor(unittest.TestCase):

    def test_validate_allowed_command(self):
        routine = {
            "description": "Test routine",
            "triggers": ["test"],
            "actions": [
                {"type": "command", "argv": ["echo", "hello"]}
            ]
        }
        errors = _validate(routine)
        self.assertEqual(len(errors), 0)

    def test_validate_blocked_command(self):
        routine = {
            "description": "Test routine",
            "triggers": ["test"],
            "actions": [
                {"type": "command", "argv": ["rm", "-rf", "/"]}
            ]
        }
        errors = _validate(routine)
        self.assertEqual(len(errors), 1)
        self.assertIn("must be an allowed command", errors[0])
        self.assertIn("found 'rm'", errors[0])

    @patch('tools.automated_routine_executor.spawn_detached')
    def test_run_action_allowed_command(self, spawn_detached):
        handle = MagicMock()
        handle.poll.return_value = 0
        handle.process.poll.return_value = 0
        spawn_detached.return_value = handle

        action = {"type": "command", "argv": ["echo", "hello"]}
        result = _run_action(action)
        self.assertTrue(result["ok"])
        self.assertEqual(result["argv"], ["echo", "hello"])
        spawn_detached.assert_called_once()

    def test_cancelled_command_stops_only_its_owned_process_tree(self):
        token = CancellationToken()
        handle = MagicMock()

        def poll():
            token.cancel("stop routine")
            return None

        handle.poll.side_effect = poll
        with (
            patch('tools.automated_routine_executor.spawn_detached', return_value=handle),
            patch('tools.automated_routine_executor.terminate_process_tree') as terminate,
            self.assertRaises(OperationCancelled),
        ):
            _run_action(
                {"type": "command", "argv": ["echo", "hello"]},
                token,
            )
        terminate.assert_called_once_with(handle)

    def test_cancelled_command_does_not_claim_unconfirmed_termination(self):
        token = CancellationToken()
        handle = MagicMock()

        def poll():
            token.cancel("stop routine")
            return None

        handle.poll.side_effect = poll
        with (
            patch('tools.automated_routine_executor.spawn_detached', return_value=handle),
            patch('tools.automated_routine_executor.terminate_process_tree', return_value=False),
            self.assertRaisesRegex(RuntimeError, "could not be confirmed"),
        ):
            _run_action(
                {"type": "command", "argv": ["echo", "hello"]},
                token,
            )

    def test_run_action_blocked_command(self):
        action = {"type": "command", "argv": ["sh", "-c", "malicious_code"]}
        with self.assertRaises(ValueError) as context:
            _run_action(action)
        self.assertIn("Command 'sh' is not permitted", str(context.exception))

if __name__ == '__main__':
    unittest.main()
