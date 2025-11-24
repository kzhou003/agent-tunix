"""Fast tests for reward functions."""

import pytest
from agent_tunix.rewards import check_answer, match_format_exactly
from agent_tunix.data import REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END


class TestCheckAnswer:
    """Test answer checking reward function."""

    def test_exact_match(self):
        """Test exact answer matching."""
        prompt = "What is 40+2?"
        completion = f"{REASONING_START}I will add 40+2{REASONING_END}{SOLUTION_START}42{SOLUTION_END}"
        answer = "42"
        reward = check_answer([prompt], [completion], [answer])
        assert len(reward) == 1 and reward[0] == 3.0, "Should give full reward for exact match"

    def test_no_match(self):
        """Test non-matching answer."""
        prompt = "What is 40+2?"
        completion = f"{REASONING_START}I will add 40+2{REASONING_END}{SOLUTION_START}10{SOLUTION_END}"
        answer = "42"
        reward = check_answer([prompt], [completion], [answer])
        assert len(reward) == 1 and reward[0] == -1.0, "Should penalize wrong answer"

    def test_partial_match_within_tolerance(self):
        """Test partial match within 10% tolerance."""
        prompt = "Estimate 40+2?"
        completion = f"{REASONING_START}close estimate{REASONING_END}{SOLUTION_START}42{SOLUTION_END}"
        answer = "40"
        reward = check_answer([prompt], [completion], [answer])
        # 42 vs 40: ratio = 1.05, within 0.9-1.1 range
        assert len(reward) == 1 and reward[0] == 0.5, "Should give partial reward for close answer"

    def test_no_solution_markers(self):
        """Test when solution markers are missing."""
        prompt = "What is the answer?"
        completion = "The answer is 42 but no markers"
        answer = "42"
        reward = check_answer([prompt], [completion], [answer])
        # Should get 0.0 since no solution markers found
        assert len(reward) == 1 and reward[0] == 0.0, "Should give no reward when markers missing"

    def test_whitespace_tolerance(self):
        """Test handling whitespace in answers."""
        prompt = "What is the answer?"
        completion = f"{REASONING_START}thinking{REASONING_END}{SOLUTION_START}  42  {SOLUTION_END}"
        answer = "42"
        reward = check_answer([prompt], [completion], [answer])
        # Should strip whitespace and match
        assert len(reward) == 1 and reward[0] == 1.5, "Should strip whitespace and get partial reward"

    def test_multiple_answers_batch(self):
        """Test processing multiple answers in batch."""
        prompts = ["Q1?", "Q2?", "Q3?"]
        completions = [
            f"{REASONING_START}r1{REASONING_END}{SOLUTION_START}10{SOLUTION_END}",
            f"{REASONING_START}r2{REASONING_END}{SOLUTION_START}20{SOLUTION_END}",
            f"{REASONING_START}r3{REASONING_END}{SOLUTION_START}30{SOLUTION_END}",
        ]
        answers = ["10", "20", "25"]
        rewards = check_answer(prompts, completions, answers)
        assert len(rewards) == 3, "Should return same number of rewards as inputs"
        assert rewards[0] == 3.0, "First should be exact match"
        assert rewards[1] == 3.0, "Second should be exact match"
        assert rewards[2] == 0.25, "Third should be partial (30 vs 25, ratio 1.2 within 0.8-1.2)"

    def test_numeric_ratio_slightly_outside_tolerance(self):
        """Test numeric answers slightly outside tolerance."""
        prompt = "Estimate?"
        completion = f"{REASONING_START}x{REASONING_END}{SOLUTION_START}50{SOLUTION_END}"
        answer = "40"
        reward = check_answer([prompt], [completion], [answer])
        # 50/40 = 1.25, outside 0.9-1.1 but within 0.8-1.2, so gets 0.25 plus base (but ratio > 1.2)
        # Actually: 1.25 > 1.2, so outside 0.8-1.2 range, gets -1.0 penalty
        assert len(reward) == 1 and reward[0] == -1.0, "Should penalize answer outside tolerance"

    def test_non_numeric_format_error(self):
        """Test when answer is not numeric."""
        prompt = "What is the color?"
        completion = f"{REASONING_START}x{REASONING_END}{SOLUTION_START}blue{SOLUTION_END}"
        answer = "red"
        reward = check_answer([prompt], [completion], [answer])
        # Neither can be parsed as float, should penalize
        assert len(reward) == 1 and reward[0] == -0.5, "Should penalize parse errors"


class TestMatchFormatExactly:
    """Test format matching reward function."""

    def test_exact_format_match(self):
        """Test exact format match with markers."""
        prompt = "Answer the question"
        completion = f"{REASONING_START}thinking here{REASONING_END}{SOLUTION_START}42{SOLUTION_END}"
        rewards = match_format_exactly([prompt], [completion])
        assert len(rewards) == 1 and rewards[0] == 3.0, "Should give full reward for exact format match"

    def test_format_no_match_missing_end(self):
        """Test format that doesn't match - missing end markers."""
        prompt = "Answer"
        completion = f"{REASONING_START}thinking{REASONING_END}{SOLUTION_START}42"  # Missing SOLUTION_END
        rewards = match_format_exactly([prompt], [completion])
        assert len(rewards) == 1 and rewards[0] == 0.0, "Should give no reward for format mismatch"

    def test_format_no_match_missing_reasoning(self):
        """Test format missing reasoning section."""
        prompt = "Answer"
        completion = f"{SOLUTION_START}42{SOLUTION_END}"
        rewards = match_format_exactly([prompt], [completion])
        assert len(rewards) == 1 and rewards[0] == 0.0, "Should give no reward without reasoning"

    def test_format_multiple_batch(self):
        """Test format matching on multiple items."""
        prompts = ["Q1", "Q2", "Q3"]
        completions = [
            f"{REASONING_START}r1{REASONING_END}{SOLUTION_START}10{SOLUTION_END}",  # Correct
            f"{REASONING_START}r2{REASONING_END}{SOLUTION_START}20",  # Missing end
            f"no markers{SOLUTION_START}30{SOLUTION_END}",  # Missing reasoning
        ]
        rewards = match_format_exactly(prompts, completions)
        assert len(rewards) == 3, "Should return 3 rewards"
        assert rewards[0] == 3.0, "First should match"
        assert rewards[1] == 0.0, "Second should not match"
        assert rewards[2] == 0.0, "Third should not match"

    def test_format_with_whitespace(self):
        """Test format matching with extra whitespace."""
        prompt = "Q?"
        completion = f"  {REASONING_START}thinking{REASONING_END}{SOLUTION_START}42{SOLUTION_END}  "
        rewards = match_format_exactly([prompt], [completion])
        # Regex allows up to 0 spaces at start/end with {0,}, should still match
        assert len(rewards) == 1 and rewards[0] == 3.0, "Should handle whitespace"

    def test_format_multiline_content(self):
        """Test format with multiline reasoning."""
        prompt = "Q?"
        completion = f"{REASONING_START}line1\nline2\nline3{REASONING_END}{SOLUTION_START}answer{SOLUTION_END}"
        rewards = match_format_exactly([prompt], [completion])
        assert len(rewards) == 1 and rewards[0] == 3.0, "Should handle multiline content"


class TestRewardRanges:
    """Test that rewards stay in valid ranges."""

    def test_check_answer_in_valid_range(self):
        """Test check_answer always returns numeric rewards."""
        test_cases = [
            ([f"{REASONING_START}r{REASONING_END}{SOLUTION_START}42{SOLUTION_END}"], ["42"]),
            ([f"{REASONING_START}r{REASONING_END}{SOLUTION_START}41{SOLUTION_END}"], ["42"]),
            ([f"{REASONING_START}r{REASONING_END}{SOLUTION_START}100{SOLUTION_END}"], ["42"]),
            ([f"{REASONING_START}r{REASONING_END}{SOLUTION_START}{SOLUTION_END}"], ["42"]),
        ]

        for completion, answer in test_cases:
            reward = check_answer(["Q?"], completion, answer)
            assert isinstance(reward, list) and len(reward) == 1, "Should return list"
            assert isinstance(reward[0], (int, float)), f"Reward should be numeric: {reward[0]}"

    def test_match_format_in_valid_range(self):
        """Test match_format_exactly always returns valid rewards."""
        test_cases = [
            (f"{REASONING_START}r{REASONING_END}{SOLUTION_START}42{SOLUTION_END}", 3.0),
            ("no markers here", 0.0),
            (f"{REASONING_START}r{REASONING_END}", 0.0),  # Missing solution
            ("", 0.0),
        ]

        for completion, expected in test_cases:
            reward = match_format_exactly(["Q?"], [completion])
            assert len(reward) == 1, "Should return list of length 1"
            assert reward[0] == expected, f"Expected {expected}, got {reward[0]}"


class TestRewardEdgeCases:
    """Test edge cases in reward functions."""

    def test_empty_completion(self):
        """Test handling of empty completion."""
        reward = check_answer(["Q?"], [""], ["42"])
        assert len(reward) == 1 and reward[0] == 0.0, "Empty completion should get no reward"

    def test_empty_answer(self):
        """Test handling of empty answer."""
        completion = f"{REASONING_START}r{REASONING_END}{SOLUTION_START}42{SOLUTION_END}"
        reward = check_answer(["Q?"], [completion], [""])
        # Should try to extract 42 and compare with empty string, will fail to parse
        assert isinstance(reward, list) and len(reward) == 1, "Should handle gracefully"

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        completion = f"{REASONING_START}r{REASONING_END}{SOLUTION_START}999999999{SOLUTION_END}"
        reward = check_answer(["Q?"], [completion], ["999999999"])
        assert len(reward) == 1 and reward[0] == 3.0, "Should handle large numbers"

    def test_negative_numbers(self):
        """Test handling of negative numbers."""
        completion = f"{REASONING_START}r{REASONING_END}{SOLUTION_START}-42{SOLUTION_END}"
        reward = check_answer(["Q?"], [completion], ["-42"])
        assert len(reward) == 1 and reward[0] == 3.0, "Should handle negative numbers"

    def test_float_numbers(self):
        """Test handling of floating point numbers."""
        completion = f"{REASONING_START}r{REASONING_END}{SOLUTION_START}3.14159{SOLUTION_END}"
        reward = check_answer(["Q?"], [completion], ["3.14159"])
        assert len(reward) == 1 and reward[0] == 3.0, "Should handle float numbers"

    def test_numeric_ratio_far_outside_tolerance(self):
        """Test numeric answers far outside tolerance."""
        completion = f"{REASONING_START}r{REASONING_END}{SOLUTION_START}100{SOLUTION_END}"
        reward = check_answer(["Q?"], [completion], ["10"])
        # 100/10 = 10.0, way outside tolerance, should penalize
        assert len(reward) == 1 and reward[0] == -1.0, "Should penalize far off answers"
