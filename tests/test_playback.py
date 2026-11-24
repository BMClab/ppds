"""Check the sensor coordinates, pressure values and clock sent to the browser."""

import base64
import json
from html.parser import HTMLParser

import numpy as np
import pytest

from ppds import playback


class RecordingParser(HTMLParser):
    def handle_data(self, data):
        if data.startswith('{"packed":'):
            self.recording = json.loads(data)


@pytest.mark.parametrize("stance", [False, True])
def test_playback_sensor_grid_and_empty_frames(monkeypatch, stance):
    monkeypatch.setattr(playback.mo, "iframe", lambda html, **kwargs: html)
    pressure = np.zeros((4, 20, 16), dtype=np.uint16)
    pressure[1, 8, 4] = 700
    pressure[1, 12, 8] = 35
    pressure[2, 9, 8] = 128
    hz = 10 if stance else 100
    parser = RecordingParser()
    parser.feed(
        playback.pressure_player(
            pressure, hz, title="trial", stance=stance, first_frame=17
        )
    )
    config = parser.recording
    packed = base64.b64decode(config["packed"])
    offsets = np.frombuffer(packed, "<u4", count=5)
    cells = np.frombuffer(packed, "<u2", offset=offsets.nbytes).reshape(-1, 2)
    np.testing.assert_array_equal(offsets, [0, 0, 2, 3, 3])
    assert (config["frames"], config["first"], config["hz"], config["peak"]) == (
        4,
        17,
        hz,
        700,
    )
    for i, source in enumerate(pressure):
        image = np.zeros((config["height"], config["width"]), dtype=np.uint16)
        pixels, values = cells[offsets[i] : offsets[i + 1]].T
        image.flat[pixels] = values
        crop = source[6:15, 2:11]  # Two sensors of padding around all loaded cells.
        # Top-down gait view: travel is rightward, so the participant's right is down.
        np.testing.assert_array_equal(image, crop[::-1] if stance else crop.T)


def test_unloaded_clip_has_a_clock(monkeypatch):
    monkeypatch.setattr(playback.mo, "iframe", lambda html, **kwargs: html)
    parser = RecordingParser()
    parser.feed(
        playback.pressure_player(np.zeros((3, 8, 5), np.uint16), 100, title="unloaded")
    )
    config = parser.recording
    assert (config["frames"], config["width"], config["height"], config["peak"]) == (
        3,
        8,
        5,
        0,
    )
    assert base64.b64decode(config["packed"]) == bytes(4 * 4)


def test_playback_uses_sampling_rate_from_npz(tmp_path, monkeypatch):
    path = tmp_path / "trial.npz"
    np.savez(path, pressure=np.ones((2, 4, 4), dtype=np.uint16), sampling_rate_hz=62.5)
    monkeypatch.setattr(playback.mo, "iframe", lambda html, **kwargs: html)
    parser = RecordingParser()
    with np.load(path) as trial:
        parser.feed(playback.pressure_player(trial["pressure"], trial["sampling_rate_hz"], title="trial"))
    assert parser.recording["hz"] == 62.5


@pytest.mark.parametrize(
    ("hz", "first_frame", "message"),
    [
        (True, 0, "sampling frequency"),
        (np.bool_(True), 0, "sampling frequency"),
        (np.array(True), 0, "sampling frequency"),
        (np.ma.array(100, mask=True), 0, "sampling frequency"),
        (np.complex128(100 + 1j), 0, "sampling frequency"),
        (np.timedelta64(100, "ns"), 0, "sampling frequency"),
        (100, True, "starting frame"),
        (100, np.bool_(True), "starting frame"),
    ],
)
def test_playback_rejects_invalid_clock_parameters(hz, first_frame, message):
    clip = np.ones((2, 4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match=message):
        playback.pressure_player(clip, hz, title="clip", first_frame=first_frame)
