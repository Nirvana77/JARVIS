"""Segmentation — a function from PCM frames to utterances (M3 decision 2).

A port of Mike's ``client/src/audio/segment.ts``, comments and tuning record
included, because the values below are not guesses: they were measured against
real rooms, including the noise-floor failure written up in Mike's PRD 6 §8.

There is no device API in this file, nothing asks the clock, and nothing here
knows where the samples came from: the edge's microphone loop hands it resampled
frames and a test hands it a recorded file. Both must produce the same segments
— acceptance criterion 8, and the reason this is a class with a ``push`` method
rather than something wired into a stream callback.

Everything is measured in samples and converted to milliseconds on the way out,
so a run is reproducible: a segmenter that consulted ``time.monotonic()`` would
give a different answer on a slow Pi than on the dev box.

The energy detector, in one paragraph. Each frame gets an RMS level in dBFS. A
frame counts as speech when it is ``margin_db`` above the noise floor, and the
floor follows the room whenever an utterance is *not* in progress — fast down,
very slowly up — so a fan, a keyboard or air conditioning raise it and stop
triggering, while a long sentence cannot raise it at all. A segment opens after
``start_ms`` of speech (a door closing is one frame, not five), carries
``pre_roll_ms`` of audio from before that so the first syllable survives, and
closes after ``hangover_ms`` of silence — long enough that an ordinary pause
inside a sentence does not split it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

#: Why a segment ended. Sent with it and logged by the brain.
SegmentReason = Literal[
    "silence",   # silence, long enough to mean the sentence ended
    "maximum",   # the maximum length
    "release",   # the push-to-talk control was released
    "close",     # the microphone was turned off mid-utterance
]


@dataclass(frozen=True)
class SegmenterOptions:
    sample_rate: int = 16_000
    #: Frame size for the energy detector. 20 ms is the usual VAD frame: long
    #: enough to measure, short enough that the boundary error is inaudible.
    frame_ms: int = 20
    #: Speech needed before a segment opens.
    start_ms: int = 100
    #: Silence that ends a segment — longer than the gap between clauses and
    #: shorter than the gap between sentences, so an ordinary pause does not
    #: split one.
    hangover_ms: int = 700
    #: Audio kept from before the first speech frame. The detector needs a few
    #: frames to be sure, and those frames contain the first consonant.
    pre_roll_ms: int = 300
    #: Audio kept after the last speech frame, so a trailing syllable survives.
    tail_ms: int = 250
    #: Less speech than this is not an utterance, and is thrown away without
    #: being sent anywhere: silence mistaken for speech must cost nothing.
    min_speech_ms: int = 250
    #: 15 s at 16 kHz is 480 kB — half of what the brain accepts, and more than
    #: anybody says in one breath.
    max_segment_ms: int = 15_000
    #: How far above the noise floor a frame has to be to count as speech.
    margin_db: float = 9.0
    #: The floor is clamped into this range. The lower bound matters: a
    #: digitally silent stream reads as -100 dB here, and an unclamped floor
    #: plus a margin would then call the first bit of dither speech.
    min_floor_db: float = -70.0
    max_floor_db: float = -25.0
    #: How fast the floor follows a quieter room, per frame (1 = instantly).
    floor_fall_rate: float = 0.5
    #: And a louder one. Deliberately two orders of magnitude slower: a fan
    #: starting should raise the bar over seconds, while a person talking for
    #: ten seconds must not raise it at all.
    floor_rise_rate: float = 0.004
    #: The floor's target is the quietest frame in this window, not the current
    #: frame — minimum statistics. Speech has gaps between words, so the
    #: quietest frame of any second and a half is the room, whatever is being
    #: said in it; and one stray frame cannot move the floor on its own. Mike's
    #: first version fell by half the distance to *every* quieter frame, so one
    #: frame of zeros from a dropout took the floor to its minimum, room tone
    #: became "speech", and the microphone was one continuous utterance cut
    #: into fifteen-second pieces until something happened to be quieter.
    floor_window_ms: int = 1500
    #: While a segment is open the floor may still rise toward that window
    #: minimum, at this rate. A segment that is open on room tone alone — the
    #: only way to be open with the window minimum above the floor — then closes
    #: itself within a couple of seconds instead of at max_segment_ms. A
    #: sentence does not raise it: its window minimum is the gap between words,
    #: which is the room.
    floor_rise_rate_open: float = 0.02
    #: A frame at or below this is not room tone, it is nothing — a dropout, a
    #: gap padded with zeros — and the floor ignores it.
    gap_db: float = -90.0
    #: A segment that reaches max_segment_ms is cut at the most recent pause of
    #: at least this length rather than at the length itself, and what follows
    #: the pause starts the next segment. A cut at an arbitrary sample lands
    #: inside a syllable, and whisper hears a different word.
    cut_gap_ms: int = 200
    #: Push-to-talk. Silence no longer ends the utterance — the hold delimits it
    #: and ``flush("release")`` ends it — but the detector still runs, so the
    #: silence at both ends is trimmed before Whisper sees it.
    hold: bool = False


DEFAULTS = SegmenterOptions()


@dataclass(eq=False)
class Segment:
    """One utterance, trimmed to the speech plus its margins."""

    #: 16 kHz signed 16-bit mono.
    pcm: np.ndarray
    #: Where this segment starts and ends in the stream, in milliseconds from
    #: the first sample ever pushed. Diagnostics, and what the file test
    #: asserts boundaries with.
    start_ms: int
    end_ms: int
    duration_ms: int
    #: How much of it was actually speech. A segment three seconds long with
    #: 0.3 s of speech in it is a cough with a hangover attached.
    speech_ms: int
    reason: SegmentReason
    #: What the detector believed the room was when it closed, and the loudest
    #: frame it kept. Sent with the segment and logged by the brain, because the
    #: one thing a log of transcripts could not answer was "what did the edge
    #: think it was hearing" — and the answer turned out to be "a floor of
    #: -70 dB and fifteen seconds of room tone".
    floor_db: int
    peak_db: int


def frame_level_db(frame: np.ndarray) -> float:
    """Level of one frame in dBFS, floored so digital silence is a number
    rather than ``-inf`` (which poisons every average it touches)."""
    if frame.size == 0:
        return -100.0
    samples = frame.astype(np.float64)
    rms = math.sqrt(float(np.mean(samples * samples))) / 32768.0
    return max(-100.0, 20.0 * math.log10(rms)) if rms > 0 else -100.0


def _round(value: float) -> int:
    """Round half *up*, like the JS original — Python's ``round`` rounds half to
    even, which would put ``tail_ms`` (250 / 20 = 12.5) one frame short."""
    return math.floor(value + 0.5)


class Segmenter:
    def __init__(self, options: SegmenterOptions | None = None) -> None:
        self.opts = options or DEFAULTS
        self.frame_samples = max(1, _round(self.opts.sample_rate * self.opts.frame_ms / 1000))

        self._frames: list[np.ndarray] = []      # the segment being built, pre-roll in front
        self._speech: list[bool] = []            # per-frame verdict, parallel to _frames
        self._levels: list[float] = []           # per-frame level, for the peak and the cut
        self._tail = np.zeros(0, dtype=np.int16)  # samples not yet a whole frame
        self._open = False
        self._start_frame = 0                    # where the current segment starts
        self._speech_frames = 0
        self._silence_frames = 0
        self._pending_speech_frames = 0
        self._consumed = 0                       # frames since the beginning of time
        self._last_db = -100.0
        self._recent: list[float] = []           # levels of the last floor_window_ms
        # Start at the top of the range and fall. The first frames of any
        # capture are room tone, and falling is the fast direction
        # (floor_fall_rate), so the floor finds a quiet room inside ~150 ms —
        # while the other way round means treating that room tone as speech
        # until the floor catches up, which is a segment of nothing sent to the
        # GPU every time the microphone is opened.
        self._floor_db = self.opts.max_floor_db

    # -- accessors --------------------------------------------------------

    @property
    def speaking(self) -> bool:
        """True while an utterance is being collected. Drives the edge's
        ``speaking`` message, which is the difference between a user who waits
        and a user who repeats themselves."""
        return self._open

    @property
    def level_db(self) -> float:
        return self._last_db

    @property
    def noise_floor_db(self) -> float:
        return self._floor_db

    @property
    def position_ms(self) -> int:
        """Milliseconds of audio pushed so far."""
        return self._frames_to_ms(self._consumed)

    # -- core -------------------------------------------------------------

    def _frames_to_ms(self, frames: int) -> int:
        return _round(frames * self.frame_samples * 1000 / self.opts.sample_rate)

    def _ms_to_frames(self, ms: float) -> int:
        return max(1, _round(ms / self.opts.frame_ms))

    def push(self, samples: np.ndarray | bytes) -> list[Segment]:
        """Feed audio in. Returns the segments that ended inside this push —
        usually none, occasionally one, more than one only if the caller handed
        over several seconds at once (a file, in a test)."""
        if isinstance(samples, (bytes, bytearray, memoryview)):
            samples = np.frombuffer(samples, dtype=np.int16)
        samples = np.asarray(samples, dtype=np.int16)

        # Join whatever was left over last time. Copying is cheap at these
        # sizes and the alternative is an index-arithmetic bug that only shows
        # up on the one input where a frame straddles two pushes.
        if self._tail.size:
            samples = np.concatenate([self._tail, samples])
            self._tail = np.zeros(0, dtype=np.int16)

        out: list[Segment] = []
        at = 0
        n = self.frame_samples
        while at + n <= samples.size:
            segment = self._frame(samples[at : at + n].copy())
            at += n
            if segment is not None:
                out.append(segment)
        if at < samples.size:
            self._tail = samples[at:].copy()
        return out

    def _frame(self, frame: np.ndarray) -> Segment | None:
        """One frame, and the whole state machine."""
        self._consumed += 1
        db = frame_level_db(frame)
        self._last_db = db

        # The floor follows the quietest recent frame (floor_window_ms), and it
        # is tracked whatever the verdict on this frame was, not only on frames
        # that looked quiet. An earlier version of Mike's updated it only when
        # `not is_speech`, which cannot recover from a floor that is too low:
        # every frame then looks like speech, so nothing ever updates the floor,
        # and a room with a fan in it is one continuous utterance. The version
        # after that froze it while a segment was open, which is the same trap
        # with an extra step: a floor that one dropout frame had dragged to the
        # bottom opened a segment on room tone, froze there, and stayed open to
        # the maximum, over and over.
        if db > self.opts.gap_db:
            self._recent.append(db)
            span = self._ms_to_frames(self.opts.floor_window_ms)
            if len(self._recent) > span:
                del self._recent[: len(self._recent) - span]
            target = min(self._recent)
            rate = (
                self.opts.floor_fall_rate if target < self._floor_db
                else self.opts.floor_rise_rate_open if self._open
                else self.opts.floor_rise_rate
            )
            self._floor_db = min(
                self.opts.max_floor_db,
                max(self.opts.min_floor_db, self._floor_db + rate * (target - self._floor_db)),
            )
        is_speech = db > self._floor_db + self.opts.margin_db

        self._frames.append(frame)
        self._speech.append(is_speech)
        self._levels.append(db)

        if not self._open:
            # Idle: keep only the pre-roll, so a microphone left on all
            # afternoon costs a fixed 300 ms of memory.
            keep = self._ms_to_frames(self.opts.pre_roll_ms) + self._ms_to_frames(self.opts.start_ms)
            if len(self._frames) > keep:
                drop = len(self._frames) - keep
                del self._frames[:drop]
                del self._speech[:drop]
                del self._levels[:drop]

            if not is_speech:
                self._pending_speech_frames = 0
                return None

            self._pending_speech_frames += 1
            if self._pending_speech_frames < self._ms_to_frames(self.opts.start_ms):
                return None

            # Open. The segment starts one pre-roll before the first of the
            # frames that convinced us, which is where the word began.
            self._open = True
            self._start_frame = max(
                0,
                len(self._frames)
                - self._pending_speech_frames
                - self._ms_to_frames(self.opts.pre_roll_ms),
            )
            self._speech_frames = self._pending_speech_frames
            self._silence_frames = 0
            self._pending_speech_frames = 0
            return None

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        # A segment has a maximum length. Cut here and let the next frame open a
        # new one — a sentence split in two is a worse answer than an unbounded
        # buffer only until the buffer is the problem.
        if len(self._frames) - self._start_frame >= self._ms_to_frames(self.opts.max_segment_ms):
            return self._close("maximum", self._cut_point())

        # In hold mode silence never ends the utterance: the release does. The
        # counting above still happens, because the trim at the end uses it.
        if not self.opts.hold and self._silence_frames >= self._ms_to_frames(self.opts.hangover_ms):
            return self._close("silence")
        return None

    def flush(self, reason: SegmentReason = "close") -> Segment | None:
        """End the current segment now — the push-to-talk release, or the
        microphone being turned off. Returns the segment, or ``None`` when there
        was nothing in it worth sending."""
        if not self._open:
            # Nothing was open, so there is nothing to send — and the pre-roll
            # ring is dropped rather than sent as an "utterance" of room tone.
            self._reset()
            return None
        return self._close(reason)

    def reset(self) -> None:
        """Forget everything, including the noise floor. Used when the
        microphone is reopened, because it may be a different microphone in a
        different room."""
        self._reset()
        # The top of the range, as in the constructor, and for the same reason:
        # falling is the fast direction. Starting at the bottom meant the first
        # room tone after a reopen was speech until the slow rise caught up —
        # which, with the segment then open, it never did.
        self._floor_db = self.opts.max_floor_db
        self._recent = []
        self._consumed = 0
        self._last_db = -100.0

    def _reset(self) -> None:
        self._frames = []
        self._speech = []
        self._levels = []
        self._tail = np.zeros(0, dtype=np.int16)
        self._open = False
        self._start_frame = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._pending_speech_frames = 0

    def _cut_point(self) -> int:
        """Where to cut a segment that has reached its maximum: the start of the
        most recent pause of ``cut_gap_ms``, if there is one in the second half
        of the segment; otherwise the end. Frames from the cut onward carry into
        the next segment."""
        gap = self._ms_to_frames(self.opts.cut_gap_ms)
        floor = self._start_frame + (len(self._speech) - self._start_frame) // 2
        run = 0
        for i in range(len(self._speech) - 1, floor - 1, -1):
            if self._speech[i]:
                run = 0
                continue
            run += 1
            if run >= gap:
                return i
        return len(self._speech)

    def _close(self, reason: SegmentReason, upto: int | None = None) -> Segment | None:
        """Cut the segment out of the buffered frames, trimmed at both ends.
        ``upto`` is where the segment stops; anything after it is the beginning
        of the next one and stays."""
        if upto is None:
            upto = len(self._speech)

        # First and last speech frame inside the segment, so the trim is the
        # same whether the silence at the front was pre-roll or (in hold mode) a
        # user who pressed the button and then thought about it.
        first = last = -1
        speech_frames = 0
        for i in range(self._start_frame, upto):
            if not self._speech[i]:
                continue
            if first < 0:
                first = i
            last = i
            speech_frames += 1
        speech_ms = self._frames_to_ms(speech_frames)

        segment: Segment | None = None
        if first >= 0 and speech_ms >= self.opts.min_speech_ms:
            from_ = max(self._start_frame, first - self._ms_to_frames(self.opts.pre_roll_ms))
            to = min(upto, last + 1 + self._ms_to_frames(self.opts.tail_ms))
            pcm = np.concatenate(self._frames[from_:to]) if to > from_ else np.zeros(0, dtype=np.int16)
            peak_db = max(self._levels[from_:to], default=-100.0)
            # The stream clock is "frames consumed"; the segment's own frames
            # are the last (len(self._frames)) of them, so its position is that
            # minus the distance from the end.
            end_of_stream = self._consumed
            start_ms = self._frames_to_ms(end_of_stream - (len(self._frames) - from_))
            end_ms = self._frames_to_ms(end_of_stream - (len(self._frames) - to))
            segment = Segment(
                pcm=pcm,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                speech_ms=speech_ms,
                reason=reason,
                floor_db=round(self._floor_db),
                peak_db=round(peak_db),
            )

        # What follows the cut is the next utterance, already in progress: it
        # reopens at once, with its own frames, rather than waiting out another
        # start_ms of speech — which would have been the first syllable, lost.
        carry_frames = self._frames[upto:]
        carry_speech = self._speech[upto:]
        carry_levels = self._levels[upto:]
        self._reset()
        if carry_frames:
            self._frames = carry_frames
            self._speech = carry_speech
            self._levels = carry_levels
            self._open = True
            self._start_frame = 0
            self._speech_frames = sum(carry_speech)
            trailing = 0
            for value in reversed(carry_speech):
                if value:
                    break
                trailing += 1
            self._silence_frames = trailing
        return segment


def segment_pcm(
    pcm: np.ndarray,
    options: SegmenterOptions | None = None,
    chunk_samples: int = 0,
) -> list[Segment]:
    """The whole of a recording, in one call.

    This is what acceptance criterion 8 is about: hand the same code a PCM file
    on the Pi and it produces the segments it produces here, because there is
    nothing else in it. It is also how the segmenter is tuned — run it over a
    recording, count the segments, look at the boundaries.
    """
    seg = Segmenter(options)
    out: list[Segment] = []
    if chunk_samples > 0:
        # Delivered in pieces, the way a sound card delivers it — including
        # pieces that do not line up with the frame size.
        for at in range(0, len(pcm), chunk_samples):
            out.extend(seg.push(pcm[at : at + chunk_samples]))
    else:
        out.extend(seg.push(pcm))
    last = seg.flush("close")
    if last is not None:
        out.append(last)
    return out


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """16-bit PCM from float32 in [-1, 1]. Its own function because it is the
    one place a sample can be silently mangled: clamping *before* scaling
    matters, since 1.0 scaled by 32768 wraps to -32768 and puts a click at the
    loudest moment of every phrase."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    scaled = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0)
    return scaled.astype(np.int16)


def resample_to(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling to the pipeline's rate, for a sound card that will not
    open at 16 kHz. Resampling happens once, on the edge, before anything else
    sees the audio.

    Linear, not windowed-sinc: the input has already been low-passed by the
    capture chain, the output goes to a model trained on telephone audio among
    other things, and the difference is inaudible to it.
    """
    if from_rate == to_rate or len(samples) == 0:
        return samples
    ratio = from_rate / to_rate
    length = int(len(samples) / ratio)
    if length <= 0:
        return np.zeros(0, dtype=np.asarray(samples).dtype)
    at = np.arange(length, dtype=np.float64) * ratio
    i0 = np.floor(at).astype(np.int64)
    i1 = np.minimum(len(samples) - 1, i0 + 1)
    t = at - i0
    src = np.asarray(samples, dtype=np.float32)
    return (src[i0] * (1 - t) + src[i1] * t).astype(np.float32)
