"""Smoke-render the video so layout regressions fail fast.

A full render is minutes of work, so nothing caught layout faults between
edits. One happened exactly that way: `captions()` counted lines but never
wrapped them, so a long line ran off both edges of the caption panel and only
showed up when a frame was decoded back out of the finished MP4.

These tests walk the whole timeline at both orientations in about a second, so
an overflow raises here instead of silently shipping.
"""

import pytest

from make_video import DURATION, Renderer, write_srt, write_timing_sheet
from video_facts import build_facts, build_scenes

ORIENTATIONS = {"vertical": (1080, 1920), "landscape": (1920, 1080)}


@pytest.fixture(scope="module")
def scenes():
    return build_scenes(build_facts())


def sample_times(scenes):
    """Every scene start, plus a dense sweep, plus each voiceover boundary."""
    times = {0.0, DURATION - 0.01}
    for scene in scenes:
        times.add(scene["start"])
        times.add(scene["end"] - 0.01)
        for start, end, _ in scene["vo"]:
            times.add(start)
            times.add(min(end - 0.01, DURATION))
    # A coarse sweep on top of the boundaries. The boundaries are where captions
    # change hands and animations peak, so they carry most of the signal; a fine
    # sweep on top of them costs minutes and finds nothing new.
    t = 0.0
    while t < DURATION:
        times.add(round(t, 3))
        t += 1.0
    return sorted(t for t in times if 0.0 <= t < DURATION)


class TestRenderSmoke:
    @pytest.mark.parametrize("name", sorted(ORIENTATIONS))
    def test_every_frame_renders(self, scenes, name):
        w, h = ORIENTATIONS[name]
        renderer = Renderer(scenes, w, h)
        for t in sample_times(scenes):
            img = renderer.render(t)
            assert img.size == (w, h)

    @pytest.mark.parametrize("name", sorted(ORIENTATIONS))
    def test_reserved_caption_box_is_sufficient(self, scenes, name):
        """No voiceover line may wrap past the space reserved for captions."""
        w, h = ORIENTATIONS[name]
        renderer = Renderer(scenes, w, h)
        f = renderer.frame
        max_w = w - 2 * f.m - 2 * f.pad
        for scene in scenes:
            for _, _, text in scene["vo"]:
                lines = f.wrap(text, f.f_cap, max_w).count("\n") + 1
                assert lines <= f.cap_lines, (
                    f"{scene['kicker']}: {lines} lines exceeds the reserved "
                    f"{f.cap_lines}: {text!r}"
                )

    def test_content_band_is_positive(self, scenes):
        for w, h in ORIENTATIONS.values():
            renderer = Renderer(scenes, w, h)
            assert renderer.frame.band_h > 0
            assert renderer.frame.band_bot <= renderer.frame.cap_top

    def test_scenes_are_contiguous_and_cover_the_timeline(self, scenes):
        assert scenes[0]["start"] == 0.0
        assert scenes[-1]["end"] == DURATION
        for earlier, later in zip(scenes, scenes[1:]):
            assert earlier["end"] == later["start"]

    def test_voiceover_stays_inside_its_scene(self, scenes):
        for scene in scenes:
            for start, end, _ in scene["vo"]:
                assert scene["start"] <= start < end <= scene["end"]


class TestGeneratedArtifacts:
    def test_srt_and_timing_sheet_write(self, scenes, tmp_path):
        facts = build_facts()
        srt = tmp_path / "out.srt"
        sheet = tmp_path / "out.md"
        write_srt(scenes, srt)
        write_timing_sheet(scenes, facts, sheet)

        text = srt.read_text(encoding="utf-8")
        assert text.count(" --> ") == sum(len(s["vo"]) for s in scenes)
        assert "-->" in text and "," in text  # comma decimal separator
        assert "Figures shown on screen" in sheet.read_text(encoding="utf-8")