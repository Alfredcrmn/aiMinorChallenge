from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from EDA.step_01_load_raw_dataframes import MOVEMENTS, SENSOR_COLUMNS, load_record
from EDA.step_02_join_all_raw_movements import load_all_movements


TIMEPOINTS = 880


def make_device_data(movement_id: int, device_id: int, records: int) -> np.ndarray:
    """Encode every source coordinate in its value so alignment is testable."""
    sample = np.arange(records)[:, None, None] * 10_000
    timepoint = np.arange(TIMEPOINTS)[None, :, None] * 10
    channel = np.arange(6)[None, None, :]
    return (movement_id * 1_000_000 + device_id * 100_000 + sample + timepoint + channel).astype(float)


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.record_counts = []
        for movement_id in range(len(MOVEMENTS)):
            records = movement_id % 3 + 1
            self.record_counts.append(records)
            for device_id in (1, 2):
                np.save(
                    self.data_dir / f"{movement_id:03d}_{device_id}.npy",
                    make_device_data(movement_id, device_id, records),
                    allow_pickle=False,
                )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_record_pairs_devices_without_changing_values(self) -> None:
        frame = load_record(5, 2, self.data_dir)

        self.assertEqual(frame.shape, (TIMEPOINTS, 17))
        np.testing.assert_array_equal(
            frame.loc[:, SENSOR_COLUMNS[:6]].to_numpy(),
            make_device_data(5, 1, 3)[2],
        )
        np.testing.assert_array_equal(
            frame.loc[:, SENSOR_COLUMNS[6:]].to_numpy(),
            make_device_data(5, 2, 3)[2],
        )
        np.testing.assert_array_equal(frame["timepoint"], np.arange(TIMEPOINTS))
        np.testing.assert_allclose(frame["time_s"], np.arange(TIMEPOINTS) / 50.0)

    def test_join_preserves_class_sample_and_timepoint_alignment(self) -> None:
        frame = load_all_movements(self.data_dir)
        expected_records = sum(self.record_counts)

        self.assertEqual(frame.shape, (expected_records * TIMEPOINTS, 17))
        self.assertEqual(frame["record_id"].nunique(), expected_records)
        self.assertEqual(frame["record_id"].min(), 0)
        self.assertEqual(frame["record_id"].max(), expected_records - 1)
        self.assertEqual(list(frame["movement_type"].cat.categories), list(MOVEMENTS))

        offset = 0
        for movement_id, records in enumerate(self.record_counts):
            movement = frame[frame["movement_type"] == MOVEMENTS[movement_id]]
            self.assertEqual(len(movement), records * TIMEPOINTS)
            self.assertEqual(movement["sample_id"].nunique(), records)
            np.testing.assert_array_equal(
                movement["record_id"].unique(),
                np.arange(offset, offset + records),
            )

            last = movement.iloc[-1]
            expected_imu = make_device_data(movement_id, 1, records)[-1, -1]
            expected_glove = make_device_data(movement_id, 2, records)[-1, -1]
            np.testing.assert_array_equal(last.loc[list(SENSOR_COLUMNS[:6])], expected_imu)
            np.testing.assert_array_equal(last.loc[list(SENSOR_COLUMNS[6:])], expected_glove)
            self.assertEqual(last["sample_id"], records - 1)
            self.assertEqual(last["timepoint"], TIMEPOINTS - 1)
            offset += records

    def test_mismatched_device_shapes_are_rejected(self) -> None:
        np.save(
            self.data_dir / "000_2.npy",
            np.zeros((2, TIMEPOINTS - 1, 6)),
            allow_pickle=False,
        )
        with self.assertRaisesRegex(ValueError, "Expected matching"):
            load_record(0, 0, self.data_dir)


if __name__ == "__main__":
    unittest.main()
