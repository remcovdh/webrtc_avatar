"""Tests for the avatar's settings: profiles, overrides and validation.

They need no GPU and no running stack, only the standard library.
"""

from dataclasses import fields
from pathlib import Path
import re
import unittest

import avatar_config
from avatar_config import AvatarSettings, ConfigError, load, variable

ROOT = Path(__file__).resolve().parent


class ProfileTests(unittest.TestCase):
    def test_every_profile_loads_and_validates(self) -> None:
        for name in avatar_config.PROFILES:
            with self.subTest(profile=name):
                loaded = load({"AVATAR_PROFILE": name})
                self.assertEqual(loaded.profile, name)
                self.assertEqual(loaded.overrides, {})
                self.assertEqual(loaded.warnings, ())

    def test_profiles_only_name_real_settings(self) -> None:
        names = {item.name for item in fields(AvatarSettings)}
        for name, changes in avatar_config.PROFILES.items():
            with self.subTest(profile=name):
                self.assertLessEqual(set(changes), names)

    def test_live_is_the_default_and_the_tested_setup(self) -> None:
        settings = load({}).settings
        self.assertEqual(load({}).profile, "live")
        self.assertEqual(settings, AvatarSettings())
        self.assertEqual(
            (settings.render_stride, settings.catchup_render_stride), (1, 2)
        )
        self.assertTrue(settings.tensorrt and settings.tts_prefetch)
        self.assertEqual(
            (settings.eye_motion_scale, settings.head_motion_scale), (0.3, 0.3)
        )
        self.assertEqual(settings.lip_sync_offset_ms, 80.0)

    def test_benchmark_profiles_differ_from_live_only_where_meant(self) -> None:
        live = load({}).settings
        runtime = avatar_config.profile_settings("benchmark-runtime")
        visual = avatar_config.profile_settings("benchmark-visual")
        self.assertFalse(runtime.tts_prefetch)
        self.assertEqual(runtime.render_stride, live.render_stride)
        self.assertTrue(runtime.incremental_frame_windows)
        self.assertFalse(visual.incremental_frame_windows)
        self.assertFalse(visual.adaptive_render_stride)
        self.assertEqual(visual.render_stride, 1)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "AVATAR_PROFILE"):
            load({"AVATAR_PROFILE": "production"})


class OverrideTests(unittest.TestCase):
    def test_override_is_applied_and_reported(self) -> None:
        loaded = load({"AVATAR_LIP_SYNC_OFFSET_MS": "120"})
        self.assertEqual(loaded.settings.lip_sync_offset_ms, 120.0)
        self.assertEqual(loaded.overrides, {"lip_sync_offset_ms": 120.0})
        self.assertEqual(
            loaded.summary()["config_overrides"], {"AVATAR_LIP_SYNC_OFFSET_MS": 120.0}
        )
        self.assertFalse(loaded.experiment)

    def test_repeating_the_profile_value_is_not_an_override(self) -> None:
        loaded = load({"AVATAR_RENDER_STRIDE": "1", "AVATAR_TENSORRT": "true"})
        self.assertEqual(loaded.overrides, {})
        self.assertEqual(loaded.config_hash, load({}).config_hash)

    def test_experiment_switch_marks_the_run(self) -> None:
        self.assertTrue(load({"AVATAR_NORMALIZE_LIP": "false"}).experiment)

    def test_hash_follows_behaviour_not_deployment(self) -> None:
        base = load({}).config_hash
        self.assertEqual(load({"AVATAR_TTS_URL": "http://tts:7860"}).config_hash, base)
        self.assertNotEqual(load({"AVATAR_RENDER_STRIDE": "2"}).config_hash, base)

    def test_keepalive_can_be_switched_off(self) -> None:
        self.assertIsNone(load({"AVATAR_AUDIO_KEEPALIVE_DBFS": "off"}).settings.audio_keepalive_dbfs)

    def test_empty_text_setting_disables_the_part(self) -> None:
        settings = load({"AVATAR_LISTENER_SOCKET": "", "AVATAR_HTTPS_PORT": ""}).settings
        self.assertEqual((settings.listener_socket, settings.https_port), ("", ""))

    def test_empty_number_means_not_set(self) -> None:
        self.assertEqual(load({"AVATAR_RENDER_STRIDE": ""}).settings.render_stride, 1)


class ValidationTests(unittest.TestCase):
    INVALID = {
        "catch-up stride below the render stride": {"AVATAR_RENDER_STRIDE": "3"},
        "stride zero": {"AVATAR_RENDER_STRIDE": "0"},
        "phrase maximum below the target": {"AVATAR_PHRASE_MAX_CHARS": "50"},
        "eye scale out of range": {"AVATAR_EYE_MOTION_SCALE": "2"},
        "lip offset out of range": {"AVATAR_LIP_SYNC_OFFSET_MS": "900"},
        "unknown lip mode": {"AVATAR_LIP_MOTION_MODE": "sideways"},
        "unknown prefetch policy": {"AVATAR_TTS_PREFETCH_POLICY": "greedy"},
        "removed voice mode": {"AVATAR_DEFAULT_VOICE_MODE": "preset-direction"},
        "not a number": {"AVATAR_RENDER_WINDOW_FRAMES": "eight"},
        "not a boolean": {"AVATAR_TENSORRT": "maybe"},
        "ICE servers not a list": {"AVATAR_ICE_SERVERS_JSON": "{}"},
        "HTTPS port not a port": {"AVATAR_HTTPS_PORT": "https"},
        "typo in a name": {"AVATAR_RENDER_STRIDES": "1"},
        "old container name": {"RENDER_STRIDE": "2"},
        "old TTS name": {"TTS_URL": "http://127.0.0.1:7860"},
        "removed flag": {"AVATAR_PASTE_BACK": "true"},
        "removed provider switch": {"TTS_PROVIDER": "breeze"},
    }

    def test_every_known_invalid_configuration_fails(self) -> None:
        for reason, environment in self.INVALID.items():
            with self.subTest(reason=reason):
                with self.assertRaises(ConfigError):
                    load(environment)

    def test_all_problems_are_reported_together(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load({"RENDER_STRIDE": "2", "AVATAR_FOO": "1"})
        message = str(caught.exception)
        self.assertIn("RENDER_STRIDE was renamed to AVATAR_RENDER_STRIDE", message)
        self.assertIn("AVATAR_FOO is not a setting", message)

    def test_fixed_stride_does_not_need_a_catchup_stride(self) -> None:
        loaded = load({
            "AVATAR_RENDER_STRIDE": "3",
            "AVATAR_ADAPTIVE_RENDER_STRIDE": "false",
        })
        self.assertEqual(loaded.settings.render_stride, 3)

    INEFFECTIVE = {
        "lip scale in absolute mode": {"AVATAR_LIP_MOTION_SCALE": "1.5"},
        "eye scale without relative motion": {
            "AVATAR_RELATIVE_MOTION": "false",
            "AVATAR_EYE_MOTION_SCALE": "0.5",
        },
        "prefetch policy with prefetch off": {
            "AVATAR_TTS_PREFETCH": "false",
            "AVATAR_TTS_PREFETCH_POLICY": "eager",
        },
        "prefetch buffer with the eager policy": {
            "AVATAR_TTS_PREFETCH_POLICY": "eager",
            "AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS": "1.0",
        },
        "catch-up stride with a fixed stride": {
            "AVATAR_ADAPTIVE_RENDER_STRIDE": "false",
            "AVATAR_CATCHUP_RENDER_STRIDE": "3",
        },
        "window size without windows": {
            "AVATAR_INCREMENTAL_FRAME_WINDOWS": "false",
            "AVATAR_RENDER_WINDOW_FRAMES": "4",
        },
        "idle frame index without the idle frame": {
            "AVATAR_USE_NEURAL_IDLE_FRAME": "false",
            "AVATAR_WARMUP_IDLE_FRAME_INDEX": "3",
        },
    }

    def test_settings_without_effect_are_warned_about(self) -> None:
        for reason, environment in self.INEFFECTIVE.items():
            with self.subTest(reason=reason):
                self.assertEqual(len(load(environment).warnings), 1)

    def test_lip_scale_is_fine_in_relative_mode(self) -> None:
        loaded = load({
            "AVATAR_LIP_MOTION_MODE": "relative",
            "AVATAR_LIP_MOTION_SCALE": "1.5",
        })
        self.assertEqual(loaded.warnings, ())


class ComposeTests(unittest.TestCase):
    """Compose must pass every setting on, and give none a value of its own."""

    @classmethod
    def setUpClass(cls) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        start = compose.index("  webrtc-avatar:\n")
        end = compose.index("    volumes:\n", start)
        cls.avatar = compose[start:end]
        cls.compose = compose

    def test_avatar_environment_is_exactly_the_settings(self) -> None:
        passed = set(re.findall(r"^      - ([A-Z0-9_]+)$", self.avatar, re.MULTILINE))
        expected = {variable(item.name) for item in fields(AvatarSettings)}
        self.assertEqual(passed, expected | {"AVATAR_PROFILE"})

    def test_compose_holds_no_defaults(self) -> None:
        self.assertFalse(":-" in self.compose, "a ${VAR:-default} is back")
        self.assertNotRegex(self.avatar, r"- [A-Z0-9_]+=")

    def test_env_example_sets_nothing(self) -> None:
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        active = [line for line in example.splitlines() if line and not line.startswith("#")]
        self.assertEqual(active, [])


if __name__ == "__main__":
    unittest.main()
