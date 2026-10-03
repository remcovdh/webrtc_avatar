"""Split a text into phrases that are spoken one after the other.

The first phrase is kept short so playback starts soon; later phrases are
synthesized and rendered while earlier ones play.
"""

from __future__ import annotations

from avatar.config import AvatarSettings


def split_phrases(text: str, settings: AvatarSettings) -> list[str]:
    """Split text into low-latency phrases without losing punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    phrases: list[str] = []
    current: list[str] = []
    current_length = 0
    closing_marks = "\"'”’)]}"

    for word in normalized.split(" "):
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)
        ending = word.rstrip(closing_marks)
        terminal = ending.endswith((".", "!", "?", ";", ":"))
        soft_break = ending.endswith(",")
        target = settings.phrase_first_target_chars if not phrases else settings.phrase_target_chars

        if (
            terminal
            or current_length >= settings.phrase_max_chars
            or current_length >= target
            or (soft_break and current_length >= max(12, target // 2))
        ):
            phrases.append(" ".join(current))
            current = []
            current_length = 0

    if current:
        tail = " ".join(current)
        if (
            phrases
            and len(tail) < 12
            and len(phrases[-1]) + 1 + len(tail) <= settings.phrase_max_chars
        ):
            phrases[-1] = f"{phrases[-1]} {tail}"
        else:
            phrases.append(tail)

    # A tiny greeting starts quickly but exhausts its media before the next
    # phrase can produce a video window. Merge it with phrase two to trade a
    # small amount of initial latency for uninterrupted opening playback.
    if (
        settings.merge_short_opening_phrase
        and len(phrases) >= 2
        and len(phrases[0]) < settings.phrase_min_first_chars
        and len(phrases[0]) + 1 + len(phrases[1]) <= settings.phrase_max_chars
    ):
        phrases[0:2] = [f"{phrases[0]} {phrases[1]}"]

    return phrases
