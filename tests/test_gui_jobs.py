import tempfile
import unittest
from pathlib import Path

from kai_local import gui_jobs
from kai_local.training import train, TrainingCancelled


class GuiJobsTests(unittest.TestCase):
    def test_argument_list_keeps_paths_as_data(self):
        args = gui_jobs.command(Path('project with spaces'), 'audit', 'vocales', [Path('capture & sample.json')], ['E', 'Nulo'])
        self.assertIn('capture & sample.json', args)
        self.assertIn('--labels', args)
        self.assertEqual(args[args.index('--category') + 1], 'vocales')

    def test_epoch_progress_only_reads_epoch_lines(self):
        self.assertEqual(gui_jobs.epoch_from_line('Epoch 12/50\n'), (12, 50))
        self.assertIsNone(gui_jobs.epoch_from_line('12/12 - 0s - loss: .5'))

    def test_cancel_before_tensorflow_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flag = root / 'cancel'
            flag.touch()
            with self.assertRaises(TrainingCancelled):
                train('vocales', 'cancel-test', ['E', 'Nulo'], {'train': []}, {'problems': []}, root / 'artifacts', cancel_file=flag)
            self.assertFalse((root / 'artifacts' / '.training.lock').exists())
            self.assertFalse((root / 'artifacts' / 'vocales' / 'cancel-test').exists())

    def test_audit_worker_reads_real_cli_response(self):
        from test_local import LocalTests
        helper = LocalTests()
        helper.setUp()
        try:
            files = helper.files()
            project = Path(__file__).resolve().parent.parent
            report = gui_jobs.run_audit(project, gui_jobs.command(project, 'audit', 'vocales', files, ['E', 'Nulo']))
            self.assertTrue(report['ready_to_train'])
            self.assertEqual(report['unique_samples'], 60)
        finally:
            helper.doCleanups()

    def test_worker_streams_complete_training_and_log(self):
        from test_local import LocalTests
        helper = LocalTests()
        helper.setUp()
        try:
            files = helper.files()
            project = Path(__file__).resolve().parent.parent
            messages = []
            args = gui_jobs.command(project, 'train', 'vocales', files, ['E', 'Nulo'], version='gui-smoke', output=helper.root / 'output', epochs=2)
            code = gui_jobs.run_training(project, args, lambda kind, line: messages.append(line), helper.root / 'worker.log')
            self.assertEqual(code, 0)
            self.assertTrue((helper.root / 'output/vocales/gui-smoke/model.kai.json').is_file())
            self.assertTrue(any(gui_jobs.epoch_from_line(line) for line in messages))
            self.assertTrue((helper.root / 'worker.log').stat().st_size > 0)
        finally:
            helper.doCleanups()

    def test_worker_cancellation_does_not_leave_a_lock(self):
        from test_local import LocalTests
        helper = LocalTests()
        helper.setUp()
        try:
            files = helper.files()
            project = Path(__file__).resolve().parent.parent
            flag = helper.root / 'cancel'
            flag.touch()
            args = gui_jobs.command(project, 'train', 'vocales', files, ['E', 'Nulo'], version='gui-cancel', output=helper.root / 'output', cancel_file=flag)
            code = gui_jobs.run_training(project, args, lambda kind, line: None, helper.root / 'cancel.log')
            self.assertEqual(code, 130)
            self.assertFalse((helper.root / 'output/.training.lock').exists())
            self.assertFalse((helper.root / 'output/vocales/gui-cancel/model.kai.json').exists())
        finally:
            helper.doCleanups()


if __name__ == '__main__':
    unittest.main()
