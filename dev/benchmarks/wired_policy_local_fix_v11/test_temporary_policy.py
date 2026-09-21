"""CPU-only tests: no sysctl, Metal, HTTP or administrator operations."""
import unittest

from dev.benchmarks.wired_policy_local_fix_v11.temporary_policy import temporary_collector_policy


class TemporaryPolicyTests(unittest.TestCase):
    def fake(self, initial=0):
        settings = {"value": initial, "writes": []}
        def read(key):
            return settings["value"]
        def write(key, value):
            settings["writes"].append(value)
            settings["value"] = value
        return settings, read, write

    def test_restores_success(self):
        settings, read, write = self.fake()
        state = {}
        with temporary_collector_policy(read, write, state):
            self.assertEqual(settings["value"], 1)
        self.assertEqual(settings["writes"], [1, 0])
        self.assertTrue(state["restored"])

    def test_restores_child_failure(self):
        settings, read, write = self.fake()
        with self.assertRaisesRegex(RuntimeError, "child failure"):
            with temporary_collector_policy(read, write):
                raise RuntimeError("child failure")
        self.assertEqual(settings["value"], 0)

    def test_restores_keyboard_interrupt(self):
        settings, read, write = self.fake()
        with self.assertRaises(KeyboardInterrupt):
            with temporary_collector_policy(read, write):
                raise KeyboardInterrupt()
        self.assertEqual(settings["writes"], [1, 0])

    def test_refuses_already_disabled(self):
        settings, read, write = self.fake(1)
        with self.assertRaisesRegex(RuntimeError, "already disabled"):
            with temporary_collector_policy(read, write):
                self.fail("must not run")
        self.assertEqual(settings["writes"], [])

    def test_refuses_unknown_initial_value(self):
        settings, read, write = self.fake(2)
        with self.assertRaises(ValueError):
            with temporary_collector_policy(read, write):
                self.fail("must not run")
        self.assertEqual(settings["writes"], [])

    def test_restores_applied_write_readback_error(self):
        settings, read, write = self.fake()
        def partial_write(key, value):
            write(key, value)
            if value == 1:
                raise RuntimeError("readback failure")
        with self.assertRaisesRegex(RuntimeError, "readback failure"):
            with temporary_collector_policy(read, partial_write):
                self.fail("must not run")
        self.assertEqual(settings["writes"], [1, 0])

    def test_concurrent_restore_is_respected(self):
        settings, read, write = self.fake()
        state = {}
        with temporary_collector_policy(read, write, state):
            settings["value"] = 0
        self.assertEqual(settings["writes"], [1])
        self.assertTrue(state["restored"])

    def test_concurrent_unknown_value_is_not_overwritten(self):
        settings, read, write = self.fake()
        state = {}
        with self.assertRaisesRegex(RuntimeError, "Concurrent policy change"):
            with temporary_collector_policy(read, write, state):
                settings["value"] = 3
        self.assertEqual(settings["writes"], [1])
        self.assertEqual(state["restore_conflict_value"], 3)


if __name__ == "__main__":
    unittest.main()
