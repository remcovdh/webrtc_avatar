"""The avatar service's settings: one source of defaults, names and rules.

Every setting is a field of `AvatarSettings`; its environment variable is
`AVATAR_` plus the field name in capitals (`render_stride` ->
`AVATAR_RENDER_STRIDE`), the same on the host and in the container. The field
defaults are the tested live setup. A profile (`AVATAR_PROFILE`) replaces some
of them, and environment variables override the profile. `load()` rejects
unknown names, values out of range and impossible combinations at start-up
instead of correcting them silently.

Only the standard library is used, so this module also runs on the host:

    python3 src/avatar/config.py show            # effective settings as JSON
    python3 src/avatar/config.py get https_port  # one value (used by entrypoint.sh)
    python3 src/avatar/config.py profiles        # the profiles as JSON

CONFIG.md describes each setting.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping

PREFIX = "AVATAR_"
PROFILE_VARIABLE = "AVATAR_PROFILE"
DEFAULT_PROFILE = "live"

# Field classes (see CONFIG.md). `profile` and `experiment` settings change
# behaviour and are part of the configuration hash; `deployment` settings say
# where things are and are not.
PROFILE = "profile"
EXPERIMENT = "experiment"
DEPLOYMENT = "deployment"


class ConfigError(ValueError):
    """The configuration cannot be used; the message lists every problem."""


def _setting(default: Any, kind: str, **rules: Any) -> Any:
    return field(default=default, metadata={"kind": kind, **rules})


@dataclass(frozen=True)
class AvatarSettings:
    # --- Rendering and timing -------------------------------------------
    # Render every Nth of JoyVASA's 25 motion frames per second.
    render_stride: int = _setting(1, PROFILE, minimum=1)
    # Switch to the catch-up stride while little media is buffered.
    adaptive_render_stride: bool = _setting(True, PROFILE)
    catchup_render_stride: int = _setting(2, PROFILE, minimum=1)
    catchup_buffer_seconds: float = _setting(0.75, PROFILE, minimum=0.0)
    # Publish rendered frames per window instead of per whole phrase.
    incremental_frame_windows: bool = _setting(True, PROFILE)
    render_window_frames: int = _setting(8, PROFILE, minimum=1)
    # Hold a phrase's speech until enough video exists to stay in sync.
    speech_start_gate: bool = _setting(True, PROFILE)
    speech_start_safety: float = _setting(1.15, PROFILE, minimum=1.0)
    # Rendered frames per second assumed until the first measurement.
    expected_render_fps: float = _setting(9.0, PROFILE, minimum=0.1)
    # Render warping_spade through TensorRT FP16 (~27 ms vs ~97 ms per frame).
    tensorrt: bool = _setting(True, PROFILE)

    # --- TTS scheduling and phrases -------------------------------------
    # Synthesize the next phrase while the current one renders.
    tts_prefetch: bool = _setting(True, PROFILE)
    tts_prefetch_policy: str = _setting(
        "adaptive", PROFILE, choices=("adaptive", "eager")
    )
    tts_prefetch_min_buffer_seconds: float = _setting(0.50, PROFILE, minimum=0.0)
    phrase_first_target_chars: int = _setting(48, PROFILE, minimum=8)
    phrase_min_first_chars: int = _setting(24, PROFILE, minimum=1)
    phrase_target_chars: int = _setting(100, PROFILE, minimum=16)
    phrase_max_chars: int = _setting(160, PROFILE, minimum=16)
    merge_short_opening_phrase: bool = _setting(True, PROFILE)
    # "design" = Chatterbox's built-in voice, "preset-clone" = the reference.
    default_voice_mode: str = _setting(
        "preset-clone", PROFILE, choices=("design", "preset-clone")
    )

    # --- Facial motion --------------------------------------------------
    eye_motion_scale: float = _setting(0.3, PROFILE, minimum=0.0, maximum=1.5)
    head_motion_scale: float = _setting(0.3, PROFILE, minimum=0.0, maximum=1.5)
    lip_motion_mode: str = _setting(
        "absolute", PROFILE, choices=("absolute", "relative")
    )
    lip_motion_scale: float = _setting(1.0, PROFILE, minimum=0.0, maximum=2.0)
    # Positive shows the mouth later than the voice.
    lip_sync_offset_ms: float = _setting(80.0, PROFILE, minimum=-500.0, maximum=500.0)
    # Level of the inaudible noise sent between sentences; "off" sends silence.
    audio_keepalive_dbfs: float | None = _setting(-60.0, PROFILE, maximum=0.0)
    # Save every phrase's WAV and JoyVASA motion here; empty disables it.
    debug_dump_dir: str = _setting("", PROFILE)

    # --- Diagnosis and rollback switches --------------------------------
    persistent_phrase_motion: bool = _setting(True, EXPERIMENT)
    relative_motion: bool = _setting(True, EXPERIMENT)
    animation_region: str = _setting(
        "all", EXPERIMENT, choices=("all", "exp", "pose", "lip", "eyes")
    )
    driving_multiplier: float = _setting(1.0, EXPERIMENT, minimum=0.0, maximum=2.0)
    normalize_lip: bool = _setting(True, EXPERIMENT)
    eye_retargeting: bool = _setting(False, EXPERIMENT)
    lip_retargeting: bool = _setting(False, EXPERIMENT)
    crop_rotation: bool = _setting(False, EXPERIMENT)
    use_neural_idle_frame: bool = _setting(True, EXPERIMENT)
    warmup_idle_frame_index: int = _setting(0, EXPERIMENT, minimum=0)

    # --- Deployment -----------------------------------------------------
    image_path: str = _setting("/workspace/inputs/avatar.jpg", DEPLOYMENT)
    preset_audio_path: str = _setting(
        "/workspace/inputs/voice-preset-nohello.wav", DEPLOYMENT
    )
    flp_config_path: str = _setting(
        "/workspace/FasterLivePortrait/configs/onnx_infer.yaml", DEPLOYMENT
    )
    results_root: str = _setting("/workspace/results", DEPLOYMENT)
    tensorrt_cache: str = _setting("/workspace/trt-cache", DEPLOYMENT)
    tts_url: str = _setting("http://127.0.0.1:7860", DEPLOYMENT)
    # Unix sockets of the listener and conductor; empty disables the part.
    listener_socket: str = _setting("/run/avatar/listener.sock", DEPLOYMENT)
    conductor_socket: str = _setting("/run/avatar/conductor.sock", DEPLOYMENT)
    port: int = _setting(8000, DEPLOYMENT, minimum=1, maximum=65535)
    # Empty disables HTTPS (and so the microphone from other machines).
    https_port: str = _setting("8443", DEPLOYMENT)
    https_cert_dir: str = _setting("/workspace/certs", DEPLOYMENT)
    # Space-separated names/addresses for the certificate; empty = this host's.
    https_cert_hosts: str = _setting("", DEPLOYMENT)
    # TURN/STUN servers for remote clients, as a JSON array.
    ice_servers_json: str = _setting("[]", DEPLOYMENT)
    log_level: str = _setting("INFO", DEPLOYMENT)


# What each profile changes relative to the field defaults above.
PROFILES: dict[str, dict[str, Any]] = {
    # The tested real-time setup; the defaults are this profile.
    "live": {},
    # Quick starts while working on the code: no TensorRT engine build, and
    # every phrase's audio and motion is kept for offline analysis.
    "development": {
        "tensorrt": False,
        "debug_dump_dir": "/workspace/results/motion-dumps",
    },
    # Benchmarks run phrases serially so timings are comparable between runs.
    # `runtime` measures the live render path, `visual` renders every frame of
    # a whole phrase before playing it, to judge mouth quality without
    # real-time pressure.
    "benchmark-runtime": {"tts_prefetch": False},
    "benchmark-visual": {
        "tts_prefetch": False,
        "render_stride": 1,
        "adaptive_render_stride": False,
        "incremental_frame_windows": False,
    },
}

# Names used before the settings got one name each. A stale .env or script
# must fail loudly rather than be ignored.
RENAMED = {
    "RENDER_STRIDE": "AVATAR_RENDER_STRIDE",
    "ADAPTIVE_RENDER_STRIDE": "AVATAR_ADAPTIVE_RENDER_STRIDE",
    "CATCHUP_RENDER_STRIDE": "AVATAR_CATCHUP_RENDER_STRIDE",
    "CATCHUP_BUFFER_SECONDS": "AVATAR_CATCHUP_BUFFER_SECONDS",
    "INCREMENTAL_FRAME_WINDOWS": "AVATAR_INCREMENTAL_FRAME_WINDOWS",
    "RENDER_WINDOW_FRAMES": "AVATAR_RENDER_WINDOW_FRAMES",
    "SPEECH_START_GATE": "AVATAR_SPEECH_START_GATE",
    "SPEECH_START_SAFETY": "AVATAR_SPEECH_START_SAFETY",
    "EXPECTED_RENDER_FPS": "AVATAR_EXPECTED_RENDER_FPS",
    "TTS_PREFETCH": "AVATAR_TTS_PREFETCH",
    "TTS_PREFETCH_POLICY": "AVATAR_TTS_PREFETCH_POLICY",
    "TTS_PREFETCH_MIN_BUFFER_SECONDS": "AVATAR_TTS_PREFETCH_MIN_BUFFER_SECONDS",
    "PHRASE_FIRST_TARGET_CHARS": "AVATAR_PHRASE_FIRST_TARGET_CHARS",
    "PHRASE_MIN_FIRST_CHARS": "AVATAR_PHRASE_MIN_FIRST_CHARS",
    "PHRASE_TARGET_CHARS": "AVATAR_PHRASE_TARGET_CHARS",
    "PHRASE_MAX_CHARS": "AVATAR_PHRASE_MAX_CHARS",
    "MERGE_SHORT_OPENING_PHRASE": "AVATAR_MERGE_SHORT_OPENING_PHRASE",
    "PERSISTENT_PHRASE_MOTION": "AVATAR_PERSISTENT_PHRASE_MOTION",
    "USE_NEURAL_IDLE_FRAME": "AVATAR_USE_NEURAL_IDLE_FRAME",
    "WARMUP_IDLE_FRAME_INDEX": "AVATAR_WARMUP_IDLE_FRAME_INDEX",
    "TTS_DEFAULT_VOICE_MODE": "AVATAR_DEFAULT_VOICE_MODE",
    "TTS_PRESET_AUDIO_PATH": "AVATAR_PRESET_AUDIO_PATH",
    "BREEZE_PRESET_AUDIO_PATH": "AVATAR_PRESET_AUDIO_PATH",
    "TTS_URL": "AVATAR_TTS_URL",
    "AVATAR_PATH": "AVATAR_IMAGE_PATH",
    "FLP_CONFIG_PATH": "AVATAR_FLP_CONFIG_PATH",
    "RESULTS_ROOT": "AVATAR_RESULTS_ROOT",
    "LISTENER_SOCKET": "AVATAR_LISTENER_SOCKET",
    "CONDUCTOR_SOCKET": "AVATAR_CONDUCTOR_SOCKET",
    "HTTPS_PORT": "AVATAR_HTTPS_PORT",
    "HTTPS_CERT_DIR": "AVATAR_HTTPS_CERT_DIR",
    "HTTPS_CERT_HOSTS": "AVATAR_HTTPS_CERT_HOSTS",
    "ICE_SERVERS_JSON": "AVATAR_ICE_SERVERS_JSON",
    "AVATAR_DEBUG_DUMP_DIR": "AVATAR_DEBUG_DUMP_DIR",
}
RENAMED = {old: new for old, new in RENAMED.items() if old != new}
# Removed settings; naming them is an error too, so nobody assumes they work.
REMOVED = (
    "AVATAR_PASTE_BACK",
    "DIRECT_MEMORY_RENDER",
    "PROGRESSIVE_PHRASE_MODE",
    "STARTUP_WARMUP",
    "TTS_PROVIDER",
    "AVATAR_PRESET_TRANSCRIPT_FILE",
)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def variable(name: str) -> str:
    """The environment variable of a field."""
    return PREFIX + name.upper()


def _parse(item: Any, raw: str) -> Any:
    text = raw.strip()
    kind = item.type
    if kind == "bool":
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ValueError("must be true or false")
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "float | None":
        return None if text.lower() in {"off", "none"} else float(text)
    return text.lower() if "choices" in item.metadata else text


@dataclass(frozen=True)
class LoadedSettings:
    """Settings plus where they came from, for /health, logs and benchmarks."""

    settings: AvatarSettings
    profile: str
    # Field -> value for everything the environment changed against the profile.
    overrides: dict[str, Any]
    # Settings that were set but cannot have any effect in this combination.
    warnings: tuple[str, ...]

    @property
    def experiment(self) -> bool:
        """True when a diagnosis/rollback switch differs from the profile."""
        kinds = {item.name: item.metadata["kind"] for item in fields(AvatarSettings)}
        return any(kinds[name] == EXPERIMENT for name in self.overrides)

    @property
    def config_hash(self) -> str:
        """Identifies the behaviour: equal hashes mean equal render, timing and
        motion settings, wherever the files and sockets are."""
        values = asdict(self.settings)
        behaviour = {
            item.name: values[item.name]
            for item in fields(AvatarSettings)
            if item.metadata["kind"] != DEPLOYMENT
        }
        encoded = json.dumps(behaviour, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "config_hash": self.config_hash,
            "config_overrides": {
                variable(name): value for name, value in self.overrides.items()
            },
            "config_experiment": self.experiment,
            "config_warnings": list(self.warnings),
        }


def _check(settings: AvatarSettings) -> list[str]:
    """Problems that make the configuration unusable."""
    problems: list[str] = []
    for item in fields(AvatarSettings):
        value = getattr(settings, item.name)
        rules = item.metadata
        name = variable(item.name)
        if value is None:
            continue
        if "choices" in rules and value not in rules["choices"]:
            problems.append(f"{name}={value!r} must be one of {', '.join(rules['choices'])}")
        if "minimum" in rules and value < rules["minimum"]:
            problems.append(f"{name}={value} must be at least {rules['minimum']}")
        if "maximum" in rules and value > rules["maximum"]:
            problems.append(f"{name}={value} must be at most {rules['maximum']}")
    if (
        settings.adaptive_render_stride
        and settings.catchup_render_stride < settings.render_stride
    ):
        problems.append(
            f"AVATAR_CATCHUP_RENDER_STRIDE={settings.catchup_render_stride} must be "
            f"at least AVATAR_RENDER_STRIDE={settings.render_stride}"
        )
    if settings.phrase_max_chars < settings.phrase_target_chars:
        problems.append(
            f"AVATAR_PHRASE_MAX_CHARS={settings.phrase_max_chars} must be at least "
            f"AVATAR_PHRASE_TARGET_CHARS={settings.phrase_target_chars}"
        )
    if settings.https_port and not (
        settings.https_port.isdigit() and 1 <= int(settings.https_port) <= 65535
    ):
        problems.append(
            f"AVATAR_HTTPS_PORT={settings.https_port!r} must be a port number or empty"
        )
    try:
        if not isinstance(json.loads(settings.ice_servers_json), list):
            raise ValueError
    except ValueError:
        problems.append("AVATAR_ICE_SERVERS_JSON must be a JSON array")
    return problems


# (setting, the switch it depends on, the reason it has no effect).
_NEEDS: tuple[tuple[str, Any, str], ...] = (
    ("eye_motion_scale", lambda s: s.relative_motion, "AVATAR_RELATIVE_MOTION is false"),
    ("head_motion_scale", lambda s: s.relative_motion, "AVATAR_RELATIVE_MOTION is false"),
    ("lip_motion_mode", lambda s: s.relative_motion, "AVATAR_RELATIVE_MOTION is false"),
    (
        "lip_motion_scale",
        lambda s: s.relative_motion and s.lip_motion_mode == "relative",
        "it only applies with AVATAR_LIP_MOTION_MODE=relative and relative motion",
    ),
    ("tts_prefetch_policy", lambda s: s.tts_prefetch, "AVATAR_TTS_PREFETCH is false"),
    (
        "tts_prefetch_min_buffer_seconds",
        lambda s: s.tts_prefetch and s.tts_prefetch_policy == "adaptive",
        "it only applies with prefetch on and AVATAR_TTS_PREFETCH_POLICY=adaptive",
    ),
    (
        "catchup_render_stride",
        lambda s: s.adaptive_render_stride,
        "AVATAR_ADAPTIVE_RENDER_STRIDE is false",
    ),
    (
        "catchup_buffer_seconds",
        lambda s: s.adaptive_render_stride,
        "AVATAR_ADAPTIVE_RENDER_STRIDE is false",
    ),
    (
        "speech_start_gate",
        lambda s: s.incremental_frame_windows,
        "AVATAR_INCREMENTAL_FRAME_WINDOWS is false",
    ),
    (
        "speech_start_safety",
        lambda s: s.incremental_frame_windows and s.speech_start_gate,
        "it only applies with incremental windows and the speech start gate",
    ),
    (
        "render_window_frames",
        lambda s: s.incremental_frame_windows,
        "AVATAR_INCREMENTAL_FRAME_WINDOWS is false",
    ),
    (
        "warmup_idle_frame_index",
        lambda s: s.use_neural_idle_frame,
        "AVATAR_USE_NEURAL_IDLE_FRAME is false",
    ),
    (
        "phrase_min_first_chars",
        lambda s: s.merge_short_opening_phrase,
        "AVATAR_MERGE_SHORT_OPENING_PHRASE is false",
    ),
)


def load(environ: Mapping[str, str] | None = None) -> LoadedSettings:
    """Build the settings from a profile plus environment overrides."""
    environ = os.environ if environ is None else environ
    problems: list[str] = []

    for old, new in RENAMED.items():
        if old in environ:
            problems.append(f"{old} was renamed to {new}")
    for name in REMOVED:
        if name in environ:
            problems.append(f"{name} was removed (see CONFIG.md)")

    by_variable = {variable(item.name): item for item in fields(AvatarSettings)}
    known = set(by_variable) | {PROFILE_VARIABLE} | set(RENAMED) | set(REMOVED)
    for name in sorted(environ):
        if name.startswith(PREFIX) and name not in known:
            problems.append(f"{name} is not a setting (see CONFIG.md)")

    profile = (environ.get(PROFILE_VARIABLE) or DEFAULT_PROFILE).strip().lower()
    if profile not in PROFILES:
        raise ConfigError(
            f"{PROFILE_VARIABLE}={profile!r} must be one of {', '.join(PROFILES)}"
        )
    base = asdict(AvatarSettings(**PROFILES[profile]))

    values = dict(base)
    explicit: set[str] = set()
    for name, item in by_variable.items():
        raw = environ.get(name)
        # An empty value means "not set", except for text settings where empty
        # is a real choice (no socket, no dump folder, no HTTPS).
        if raw is None or (raw.strip() == "" and item.type != "str"):
            continue
        try:
            values[item.name] = _parse(item, raw)
        except ValueError as exc:
            reason = str(exc) if "must be" in str(exc) else f"is not a valid {item.type}"
            problems.append(f"{name}={raw!r} {reason}")
            continue
        explicit.add(item.name)

    settings = AvatarSettings(**values)
    if not problems:
        problems.extend(_check(settings))
    if problems:
        raise ConfigError(
            "Invalid avatar configuration:\n" + "\n".join(f"  - {p}" for p in problems)
        )

    overrides = {
        name: values[name] for name in sorted(explicit) if values[name] != base[name]
    }
    warnings = tuple(
        f"{variable(name)} is set but has no effect: {reason}"
        for name, effective, reason in _NEEDS
        if name in overrides and not effective(settings)
    )
    return LoadedSettings(settings, profile, overrides, warnings)


def profile_settings(profile: str) -> AvatarSettings:
    """A profile's settings without any environment override."""
    return load({PROFILE_VARIABLE: profile}).settings


def _main(arguments: list[str]) -> int:
    command = arguments[0] if arguments else ""
    try:
        if command == "profiles":
            print(json.dumps(PROFILES, indent=2))
            return 0
        loaded = load()
        if command == "show":
            print(json.dumps({**loaded.summary(), "settings": asdict(loaded.settings)}, indent=2))
            return 0
        if command == "get" and len(arguments) == 2:
            print(getattr(loaded.settings, arguments[1]))
            return 0
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
