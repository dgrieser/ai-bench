#!/usr/bin/env python3
"""Tests for the type-to-search prompt helpers. Run with ./test_selector.py

Two properties are easy to break and impossible to see in a diff, so they are
pinned against a small ANSI screen emulator instead:

  * a redraw must never touch a line above the prompt -- the earlier renderer
    moved the cursor up by the whole block height first, which walked the block
    up the screen and overwrote whatever had been printed before it;
  * Tab must add only the characters every remaining match agrees on, not jump
    to one of them.
"""

from __future__ import annotations

import io
import os
import re
import unittest
from contextlib import redirect_stdout
from unittest import mock

import _selector as selector

OPTIONS = ["kimi-k2-5", "kimi-k2-6", "kimi-k3", "minimax-m3"]


def render_screen(stream: str, width: int = 80, height: int = 10) -> list[str]:
    """The visible screen after a raw-mode byte stream.

    Understands only what the selector emits: \\r, \\n, erase-line (CSI 2K),
    erase-to-end-of-screen (CSI J) and cursor-up-to-column-0 (CSI nF). \\n is a
    bare line feed, as in raw mode, so column is kept.
    """
    grid = [[" "] * width for _ in range(height)]
    row = col = 0
    index = 0
    while index < len(stream):
        char = stream[index]
        if char == "\x1b":
            match = re.match(r"\x1b\[(\d*)([A-Za-z])", stream[index:])
            if match is None:
                index += 1
                continue
            count, command = int(match.group(1) or 0), match.group(2)
            if command == "K":
                start = 0 if count == 2 else col
                for column in range(start, width):
                    grid[row][column] = " "
            elif command == "J":
                for column in range(col, width):
                    grid[row][column] = " "
                for line in range(row + 1, height):
                    grid[line] = [" "] * width
            elif command == "F":
                row = max(0, row - max(1, count))
                col = 0
            index += match.end()
            continue
        if char == "\r":
            col = 0
        elif char == "\n":
            row = min(height - 1, row + 1)
        elif col < width:
            grid[row][col] = char
            col += 1
        index += 1
    return ["".join(line).rstrip() for line in grid]


class TestFindMatches(unittest.TestCase):
    def test_empty_query_offers_everything_up_to_the_limit(self) -> None:
        self.assertEqual(selector.find_matches("", OPTIONS, limit=2), OPTIONS[:2])

    def test_substring_beats_subsequence(self) -> None:
        matches = selector.find_matches("k3", ["kimi-k3", "kimi-k2-3"])
        self.assertEqual(matches[0], "kimi-k3")

    def test_no_match_is_no_result(self) -> None:
        self.assertEqual(selector.find_matches("zzz", OPTIONS), [])


class TestTabCompletion(unittest.TestCase):
    def test_stops_at_the_shared_prefix(self) -> None:
        matches = selector.find_matches("kimi", OPTIONS)
        self.assertEqual(len(matches), 3)
        self.assertEqual(selector.tab_completion("kimi", matches), "kimi-k")

    def test_never_picks_one_of_several_candidates(self) -> None:
        matches = selector.find_matches("kimi-k", OPTIONS)
        self.assertEqual(len(matches), 3)
        self.assertIsNone(selector.tab_completion("kimi-k", matches))
        self.assertIsNone(selector.tab_completion("kimi-k2-", OPTIONS[:2]))

    def test_single_match_completes_fully(self) -> None:
        self.assertEqual(selector.tab_completion("min", ["minimax-m3"]), "minimax-m3")

    def test_single_fuzzy_match_completes_fully(self) -> None:
        # "k3" is not a prefix of the match, but there is nothing else it can be.
        self.assertEqual(selector.tab_completion("k3", ["kimi-k3"]), "kimi-k3")

    def test_fuzzy_typing_with_several_matches_is_left_alone(self) -> None:
        # Growing "k2" to the shared "kimi-k2-" prefix would drop what was typed.
        self.assertIsNone(selector.tab_completion("k2", ["kimi-k2-5", "kimi-k2-6"]))

    def test_completed_value_is_not_recompleted(self) -> None:
        self.assertIsNone(selector.tab_completion("minimax-m3", ["minimax-m3"]))

    def test_casing_comes_from_the_option(self) -> None:
        self.assertEqual(
            selector.tab_completion("qwen3-c", ["Qwen3-Coder-A", "Qwen3-Coder-B"]),
            "Qwen3-Coder-",
        )

    def test_no_matches_is_no_completion(self) -> None:
        self.assertIsNone(selector.tab_completion("k", []))


class TestRendering(unittest.TestCase):
    def draw(self, keystrokes: list[str], *, clear: bool = False) -> list[str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print("above one", end="\r\n")
            print("above two", end="\r\n")
            height = 0
            for typed in keystrokes:
                height = selector.render_selector(
                    "Model", typed, selector.find_matches(typed, OPTIONS)
                )
            if clear:
                selector.clear_selector(height)
        return render_screen(buffer.getvalue())

    def test_first_draw_puts_matches_under_the_prompt(self) -> None:
        screen = self.draw(["kimi-k2-"])
        self.assertEqual(screen[:5], [
            "above one",
            "above two",
            "Model: kimi-k2-",
            "  kimi-k2-5",
            "  kimi-k2-6",
        ])

    def test_redraw_keeps_the_lines_above_the_prompt(self) -> None:
        screen = self.draw(["k", "ki", "kim", "kimi", "kimi-k", "kimi-k3"])
        self.assertEqual(screen[:4], [
            "above one",
            "above two",
            "Model: kimi-k3",
            "  kimi-k3",
        ])

    def test_shrinking_block_erases_the_rows_it_gave_up(self) -> None:
        screen = self.draw(["kimi", "minimax"])
        self.assertEqual(screen[2:6], ["Model: minimax", "  minimax-m3", "", ""])

    def test_clear_blanks_the_block_and_nothing_above(self) -> None:
        screen = self.draw(["kimi"], clear=True)
        self.assertEqual(screen[:4], ["above one", "above two", "", ""])

    def test_height_counts_the_prompt_line(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(selector.render_selector("Model", "", []), 1)
            self.assertEqual(selector.render_selector("Model", "kimi-k2-", OPTIONS[:2]), 3)

    def test_long_lines_are_truncated_instead_of_wrapping(self) -> None:
        option = "https://example.com/" + "x" * 200
        size = os.terminal_size((40, 24))
        with mock.patch.object(selector.shutil, "get_terminal_size", return_value=size):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                height = selector.render_selector("URL", "https", [option])
            screen = render_screen(buffer.getvalue(), width=40)
        self.assertEqual(height, 2)
        self.assertTrue(all(len(line) < 40 for line in screen))


if __name__ == "__main__":
    unittest.main()
