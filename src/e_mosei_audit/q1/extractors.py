"""Lazy external feature adapters for Question 1 time-aligned extraction.

MediaPipe rows end with face area, face centre x/y, and face presence. The final
channel is not a detector-confidence score: no detected face produces no row,
and every emitted face row has a presence value of ``1.0``.
"""

from __future__ import annotations

import importlib
import math
import operator
import re
import stat
import wave
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from e_mosei_audit.q1.timeline import pool_intervals


_BERT_DIMENSIONS = 768
_OPENSMILE_DIMENSIONS = 25
_MEDIAPIPE_BLENDSHAPE_DIMENSIONS = 52
_MEDIAPIPE_DIMENSIONS = 56


@dataclass(frozen=True)
class WordInterval:
    """A word from the supplied transcript with physical audio timing."""

    word: str
    start: float
    end: float
    char_start: int
    char_end: int
    aligned_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("word must be a non-empty string")
        start = _finite_time(self.start, "start")
        end = _finite_time(self.end, "end")
        if end <= start:
            raise ValueError("word time interval must be finite and positive")
        char_start = _character_offset(self.char_start, "char_start")
        char_end = _character_offset(self.char_end, "char_end")
        if char_end < char_start:
            raise ValueError("word character span must be ordered")
        if self.aligned_text is not None and (
            not isinstance(self.aligned_text, str) or not self.aligned_text.strip()
        ):
            raise ValueError("aligned_text must be a non-empty string when supplied")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "char_start", char_start)
        object.__setattr__(self, "char_end", char_end)


class WordAligner(Protocol):
    """Align the original supplied transcript to physical audio time."""

    def align(
        self, wav_path: Path, transcript: str, duration: float
    ) -> tuple[WordInterval, ...]: ...


class TextEncoder(Protocol):
    """Return token vectors and the original-word-derived timing for each token."""

    def encode(
        self, transcript: str, words: Sequence[WordInterval]
    ) -> tuple[np.ndarray, tuple[WordInterval, ...]]: ...


class AudioExtractor(Protocol):
    """Return unpooled audio windows with their physical start and end times."""

    def extract(self, wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class VisionExtractor(Protocol):
    """Return unpooled detected-face frames with their physical start and end times."""

    def extract(self, frame_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def close(self) -> None: ...


def text_slot_features(
    embeddings: object, intervals: Sequence[WordInterval], slots: object
) -> tuple[np.ndarray, np.ndarray]:
    """Pool token embeddings only by the physical intervals supplied for them."""

    values = _float_matrix(embeddings, "embeddings")
    normalized_intervals = _word_intervals(intervals)
    if values.shape[0] != len(normalized_intervals):
        raise ValueError("embeddings and intervals must have matching leading dimensions")
    starts = np.asarray([interval.start for interval in normalized_intervals])
    ends = np.asarray([interval.end for interval in normalized_intervals])
    slot_values = np.asarray(slots)
    if np.iscomplexobj(slot_values):
        raise ValueError("slots must be a real numeric array")
    return pool_intervals(values, starts, ends, slot_values)


def normalize_bert_embeddings(values: object) -> np.ndarray:
    """Validate the fixed token-level BERT representation."""

    embeddings = _float_matrix(values, "BERT embeddings")
    if embeddings.shape[1] != _BERT_DIMENSIONS:
        raise ValueError("BERT embeddings must have exactly 768 dimensions")
    return embeddings


def normalize_opensmile_windows(
    values: object, starts: object, ends: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate eGeMAPSv02 low-level descriptor windows and their timestamps."""

    windows = _float_matrix(values, "OpenSMILE windows")
    if windows.shape[1] != _OPENSMILE_DIMENSIONS:
        raise ValueError("OpenSMILE windows must have exactly 25 dimensions")
    normalized_starts, normalized_ends = _time_bounds(starts, ends, len(windows))
    return windows, normalized_starts, normalized_ends


def normalize_mediapipe_frame(
    blendshape_scores: object,
    *,
    face_area: object,
    center_x: object,
    center_y: object,
    face_presence: object,
) -> np.ndarray:
    """Build [52 blendshapes, area, centre x/y, face presence] for one face."""

    scores = np.asarray(blendshape_scores)
    if scores.ndim != 1 or np.iscomplexobj(scores):
        raise ValueError("MediaPipe blendshape scores must be a real one-dimensional array")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            scores = scores.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "MediaPipe blendshape scores must be a real one-dimensional array"
        ) from error
    if not np.all(np.isfinite(scores)):
        raise ValueError("MediaPipe blendshape scores must be finite")

    features = np.zeros(_MEDIAPIPE_DIMENSIONS, dtype=np.float32)
    features[: min(len(scores), _MEDIAPIPE_BLENDSHAPE_DIMENSIONS)] = scores[
        :_MEDIAPIPE_BLENDSHAPE_DIMENSIONS
    ]
    presence = _finite_scalar(face_presence, "face_presence")
    if presence != 1.0:
        raise ValueError("face_presence must be 1.0 for a detected face")
    with np.errstate(over="ignore", invalid="ignore"):
        features[-4:] = np.asarray(
            [
                _finite_scalar(face_area, "face_area"),
                _finite_scalar(center_x, "center_x"),
                _finite_scalar(center_y, "center_y"),
                presence,
            ],
            dtype=np.float32,
        )
    if features[-4] < 0:
        raise ValueError("face_area must be non-negative")
    if not np.all(np.isfinite(features)):
        raise ValueError("MediaPipe features must be finite after float32 conversion")
    return features


def build_whisperx_aligner(model_cache: Path) -> WordAligner:
    """Create a no-download WhisperX word aligner for the supplied cache."""

    whisperx = _optional_package("whisperx")
    cache = _model_cache(model_cache, "WhisperX")
    _cached_asset(
        cache,
        "facebook/wav2vec2-base-960h",
        "models--facebook--wav2vec2-base-960h",
        "wav2vec2-base-960h",
    )
    nltk, nltk_data_root = _local_punkt_tab(cache)
    transformers = _optional_package("transformers")
    try:
        processor = transformers.Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h",
            cache_dir=str(cache),
            local_files_only=True,
        )
        align_model = transformers.Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h",
            cache_dir=str(cache),
            local_files_only=True,
        )
        align_model = align_model.to("cpu")
        align_model.eval()
        metadata = {
            "language": "en",
            "dictionary": {
                character.lower(): code
                for character, code in processor.tokenizer.get_vocab().items()
            },
            "type": "huggingface",
        }
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "WhisperX could not load expected local alignment asset through "
            "Transformers "
            "'facebook/wav2vec2-base-960h' from model cache "
            f"{cache}; downloads are disabled"
        ) from error
    return _WhisperXAligner(whisperx, align_model, metadata, nltk, nltk_data_root)


def build_bert_encoder(model_cache: Path) -> TextEncoder:
    """Load local ``bert-base-uncased`` assets without permitting downloads."""

    transformers = _optional_package("transformers")
    torch = _optional_package("torch")
    cache = _model_cache(model_cache, "BERT")
    asset = _cached_asset(
        cache,
        "bert-base-uncased",
        "bert-base-uncased",
        "models--google-bert--bert-base-uncased",
        "models--bert-base-uncased",
    )
    source = str(asset) if (asset / "config.json").is_file() else "bert-base-uncased"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            source,
            cache_dir=str(cache),
            local_files_only=True,
            use_fast=True,
        )
        model = transformers.AutoModel.from_pretrained(
            source,
            cache_dir=str(cache),
            local_files_only=True,
        )
    except OSError as error:
        raise RuntimeError(
            "BERT expected local asset 'bert-base-uncased' in model cache "
            f"{cache}; downloads are disabled"
        ) from error
    model.eval()
    return _BertEncoder(tokenizer, model, torch)


def build_opensmile_extractor() -> AudioExtractor:
    """Create an eGeMAPSv02 low-level-descriptor extractor."""

    opensmile = _optional_package("opensmile")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    return _OpenSmileExtractor(smile)


def build_mediapipe_extractor(model_cache: Path) -> VisionExtractor:
    """Create a Face Landmarker extractor from a local task-model file."""

    mediapipe = _optional_package("mediapipe")
    cache = _model_cache(model_cache, "MediaPipe")
    asset = cache / "face_landmarker.task"
    if not asset.is_file():
        raise RuntimeError(
            "MediaPipe expected local asset 'face_landmarker.task' in model cache "
            f"{cache}; downloads are disabled"
        )
    try:
        base_options = mediapipe.tasks.BaseOptions(model_asset_path=str(asset))
        options = mediapipe.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mediapipe.tasks.vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
        )
        landmarker = mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "MediaPipe could not load expected asset 'face_landmarker.task' from "
            f"model cache {cache}"
        ) from error
    return _MediaPipeExtractor(mediapipe, landmarker)


class _WhisperXAligner:
    def __init__(
        self,
        whisperx: Any,
        align_model: Any,
        metadata: Any,
        nltk: Any,
        nltk_data_root: Path,
    ) -> None:
        self._whisperx = whisperx
        self._align_model = align_model
        self._metadata = metadata
        self._nltk = nltk
        self._nltk_data_root = nltk_data_root

    def align(
        self, wav_path: Path, transcript: str, duration: float
    ) -> tuple[WordInterval, ...]:
        _wav_file(wav_path)
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be a non-empty string")
        duration = _finite_time(duration, "duration")
        if duration <= 0:
            raise ValueError("duration must be finite and positive")
        audio = _decoded_wav_audio(wav_path)
        aligned = self._run_alignment(audio, transcript, duration)
        try:
            return _aligned_word_intervals(aligned, transcript)
        except RuntimeError as error:
            if str(error) != "WhisperX alignment returned a word without timestamps":
                raise

        fallback = _normalized_alignment_plan(transcript)
        aligned = self._run_alignment(audio, fallback.text, duration)
        return _normalized_word_intervals(aligned, transcript, fallback)

    def _run_alignment(self, audio: np.ndarray, transcript: str, duration: float) -> object:
        with _nltk_alignment_guard(self._nltk, self._nltk_data_root):
            return self._whisperx.align(
                [{"start": 0.0, "end": duration, "text": transcript}],
                self._align_model,
                self._metadata,
                audio,
                "cpu",
                return_char_alignments=False,
            )


class _BertEncoder:
    def __init__(self, tokenizer: Any, model: Any, torch: Any) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def encode(
        self, transcript: str, words: Sequence[WordInterval]
    ) -> tuple[np.ndarray, tuple[WordInterval, ...]]:
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be a non-empty string")
        source_words = _word_intervals(words)
        encoded = self._tokenizer(
            transcript,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )
        offsets = _offset_pairs(encoded.pop("offset_mapping"))
        with self._torch.no_grad():
            output = self._model(**encoded)
        embeddings = normalize_bert_embeddings(
            output.last_hidden_state[0].detach().cpu().numpy()
        )
        if len(offsets) != len(embeddings):
            raise RuntimeError("BERT tokenizer offsets do not match token embeddings")
        token_intervals = tuple(
            _token_interval(transcript, offset, source_words) for offset in offsets
        )
        return embeddings, token_intervals


class _OpenSmileExtractor:
    def __init__(self, smile: Any) -> None:
        self._smile = smile

    def extract(self, wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _wav_file(wav_path)
        frame = self._smile.process_file(str(wav_path))
        starts, ends = _opensmile_timestamps(frame.index)
        return normalize_opensmile_windows(frame.to_numpy(), starts, ends)


class _MediaPipeExtractor:
    def __init__(self, mediapipe: Any, landmarker: Any) -> None:
        self._mediapipe = mediapipe
        self._landmarker = landmarker

    def extract(self, frame_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(frame_dir, Path) or not frame_dir.is_dir():
            raise ValueError(f"frame_dir must be an existing directory: {frame_dir}")
        feature_rows: list[np.ndarray] = []
        starts: list[float] = []
        ends: list[float] = []
        for index, frame_path in enumerate(sorted(frame_dir.glob("*.png"))):
            result = self._landmarker.detect(
                self._mediapipe.Image.create_from_file(str(frame_path))
            )
            if not getattr(result, "face_landmarks", None):
                continue
            landmarks = result.face_landmarks[0]
            blendshapes = getattr(result, "face_blendshapes", ())
            scores = [category.score for category in blendshapes[0]] if blendshapes else []
            feature_rows.append(
                normalize_mediapipe_frame(
                    scores,
                    face_area=_face_area(landmarks),
                    center_x=_face_center(landmarks, "x"),
                    center_y=_face_center(landmarks, "y"),
                    face_presence=1.0,
                )
            )
            start = index / 10.0
            starts.append(start)
            ends.append(start + 0.1)
        if not feature_rows:
            return (
                np.empty((0, _MEDIAPIPE_DIMENSIONS), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        return (
            np.stack(feature_rows).astype(np.float32, copy=False),
            np.asarray(starts, dtype=np.float64),
            np.asarray(ends, dtype=np.float64),
        )

    def close(self) -> None:
        self._landmarker.close()


def _finite_time(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} time must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} time must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} time must be finite")
    return numeric


def _character_offset(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} character offset must be non-negative")
    try:
        offset = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} character offset must be non-negative") from error
    if offset < 0:
        raise ValueError(f"{name} character offset must be non-negative")
    return offset


def _finite_scalar(value: object, name: str) -> float:
    numeric = _finite_time(value, name)
    return numeric


def _float_matrix(value: object, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim != 2 or np.iscomplexobj(values):
        raise ValueError(f"{name} must be a 2D real floating-point array")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must be a 2D real floating-point array")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be finite after float32 conversion")
    return converted


def _word_intervals(intervals: Sequence[WordInterval]) -> tuple[WordInterval, ...]:
    if isinstance(intervals, (str, bytes)):
        raise ValueError("intervals must be WordInterval values")
    try:
        normalized = tuple(intervals)
    except TypeError as error:
        raise ValueError("intervals must be WordInterval values") from error
    if not all(isinstance(interval, WordInterval) for interval in normalized):
        raise ValueError("intervals must be WordInterval values")
    return normalized


def _time_bounds(
    starts: object, ends: object, expected_count: int
) -> tuple[np.ndarray, np.ndarray]:
    start_values = np.asarray(starts)
    end_values = np.asarray(ends)
    if (
        start_values.ndim != 1
        or end_values.ndim != 1
        or len(start_values) != expected_count
        or len(end_values) != expected_count
        or np.iscomplexobj(start_values)
        or np.iscomplexobj(end_values)
    ):
        raise ValueError("window starts and ends must match feature rows")
    try:
        start_values = start_values.astype(np.float64, copy=False)
        end_values = end_values.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("window starts and ends must be finite") from error
    if not np.all(np.isfinite(start_values)) or not np.all(np.isfinite(end_values)):
        raise ValueError("window starts and ends must be finite")
    if np.any(end_values <= start_values):
        raise ValueError("window timestamps must have positive duration")
    return start_values, end_values


def _optional_package(package: str) -> Any:
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name == package:
            raise RuntimeError(
                f"optional package '{package}' is required for Q1 extraction; "
                "install e-mosei-audit[q1]"
            ) from error
        raise


def _model_cache(model_cache: object, adapter: str) -> Path:
    if not isinstance(model_cache, Path) or not model_cache.is_dir():
        raise RuntimeError(
            f"{adapter} requires an existing local model cache directory: {model_cache}"
        )
    return model_cache


def _cached_asset(model_cache: Path, expected: str, *candidates: str) -> Path:
    for candidate in candidates:
        asset = model_cache / candidate
        if asset.is_dir() or asset.is_file():
            return asset
    raise RuntimeError(
        f"expected local asset '{expected}' in model cache {model_cache}; "
        "downloads are disabled"
    )


def _local_punkt_tab(model_cache: Path) -> tuple[Any, Path]:
    nltk_data_root = model_cache / "nltk_data"
    try:
        nltk = _optional_package("nltk")
    except RuntimeError as error:
        raise _punkt_tab_error(model_cache) from error

    try:
        _find_local_punkt_tab(nltk, nltk_data_root)
    except (AttributeError, LookupError, OSError, TypeError, ValueError) as error:
        raise _punkt_tab_error(model_cache) from error
    return nltk, nltk_data_root


def _find_local_punkt_tab(nltk: Any, nltk_data_root: Path) -> object:
    find = nltk.data.find
    paths = [str(nltk_data_root)]
    resource = "tokenizers/punkt_tab/english.pickle"
    try:
        return find(resource, paths=paths)
    except LookupError:
        # Current NLTK releases resolve the PunktTab language directory directly.
        return find("tokenizers/punkt_tab/english/", paths=paths)


def _punkt_tab_error(model_cache: Path) -> RuntimeError:
    return RuntimeError(
        "WhisperX requires English punkt_tab in model_cache "
        f"{model_cache / 'nltk_data'}; downloads are disabled"
    )


@contextmanager
def _nltk_alignment_guard(nltk: Any, nltk_data_root: Path):
    try:
        paths = nltk.data.path
        original_paths = list(paths)
        original_download = nltk.download
    except (AttributeError, TypeError) as error:
        raise _punkt_tab_error(nltk_data_root.parent) from error

    def blocked_download(*args: object, **kwargs: object) -> object:
        requested = args[0] if args else kwargs.get("info_or_id", "unknown resource")
        raise RuntimeError(
            "WhisperX attempted an NLTK download for "
            f"{requested!r}, including punkt_tab; downloads are disabled"
        )

    try:
        paths[:] = [str(nltk_data_root)]
        nltk.download = blocked_download
        yield
    finally:
        paths[:] = original_paths
        nltk.download = original_download


def _wav_file(wav_path: object) -> Path:
    if not isinstance(wav_path, Path) or not wav_path.is_file():
        raise ValueError(f"wav_path must be an existing file: {wav_path}")
    return wav_path


def _decoded_wav_audio(wav_path: Path) -> np.ndarray:
    """Read the configured FFmpeg WAV contract without delegating to WhisperX."""

    try:
        if (
            not isinstance(wav_path, Path)
            or not wav_path.is_file()
            or not stat.S_ISREG(wav_path.stat().st_mode)
        ):
            raise ValueError(_decoded_wav_contract())
        with wave.open(str(wav_path), "rb") as stream:
            if (
                stream.getcomptype() != "NONE"
                or stream.getnchannels() != 1
                or stream.getframerate() != 16000
                or stream.getsampwidth() != 2
            ):
                raise ValueError(_decoded_wav_contract())
            frame_width = stream.getnchannels() * stream.getsampwidth()
            data_size = _declared_wav_data_size(stream)
            if data_size <= 0 or data_size % frame_width:
                raise ValueError(_decoded_wav_contract())
            frame_count = stream.getnframes()
            if frame_count != data_size // frame_width:
                raise ValueError(_decoded_wav_contract())
            raw_audio = stream.readframes(frame_count)
    except ValueError:
        raise
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError(_decoded_wav_contract()) from error

    if len(raw_audio) != frame_count * 2:
        raise ValueError(_decoded_wav_contract())
    samples = np.frombuffer(raw_audio, dtype="<i2")
    if samples.ndim != 1 or len(samples) != frame_count:
        raise ValueError(_decoded_wav_contract())
    audio = samples.astype(np.float32) / np.float32(32768.0)
    if (
        audio.dtype != np.float32
        or not np.all(np.isfinite(audio))
        or np.any(audio < -1.0)
        or np.any(audio >= 1.0)
    ):
        raise ValueError(_decoded_wav_contract())
    return audio


def _decoded_wav_contract() -> str:
    return (
        "decoded WAV must be a readable RIFF/WAVE regular file with one channel, "
        "16 kHz signed 16-bit PCM, and nonempty integral frames"
    )


def _declared_wav_data_size(stream: Any) -> int:
    """Return Wave_read's raw data-chunk length before frame-count flooring."""

    try:
        data_size = stream._data_chunk.chunksize
    except AttributeError as error:
        raise ValueError(_decoded_wav_contract()) from error
    if isinstance(data_size, bool) or not isinstance(data_size, int):
        raise ValueError(_decoded_wav_contract())
    return data_size


def _aligned_word_intervals(aligned: object, transcript: str) -> tuple[WordInterval, ...]:
    word_rows = _timestamped_words(aligned)
    result: list[WordInterval] = []
    cursor = 0
    for word, start, end in word_rows:
        char_start, char_end = _transcript_span(transcript, word, cursor)
        cursor = char_end
        result.append(
            WordInterval(
                word=word,
                start=start,
                end=end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _AlignmentSpan:
    char_start: int
    char_end: int
    alignment_words: tuple[str, ...]


@dataclass(frozen=True)
class _NormalizedAlignment:
    text: str
    spans: tuple[_AlignmentSpan, ...]


def _normalized_alignment_plan(transcript: str) -> _NormalizedAlignment:
    spans: list[_AlignmentSpan] = []
    pending_start: int | None = None
    for match in re.finditer(r"\S+", transcript):
        words = _alignment_words(match.group())
        if not words:
            if spans:
                previous = spans[-1]
                spans[-1] = _AlignmentSpan(
                    char_start=previous.char_start,
                    char_end=match.end(),
                    alignment_words=previous.alignment_words,
                )
            elif pending_start is None:
                pending_start = match.start()
            continue
        spans.append(
            _AlignmentSpan(
                char_start=pending_start if pending_start is not None else match.start(),
                char_end=match.end(),
                alignment_words=words,
            )
        )
        pending_start = None
    if not spans:
        raise RuntimeError("WhisperX alignment normalization removed all spoken text")
    return _NormalizedAlignment(
        text=" ".join(word for span in spans for word in span.alignment_words),
        spans=tuple(spans),
    )


def _alignment_words(source: str) -> tuple[str, ...]:
    normalized = source.translate(
        str.maketrans(
            {
                "‘": "'",
                "’": "'",
                "–": " ",
                "—": " ",
                "“": " ",
                "”": " ",
                "…": " ",
            }
        )
    )
    numeric = re.fullmatch(
        r"([0-9][0-9,]*)(st|nd|rd|th)?[.,;:!?]*", normalized, flags=re.IGNORECASE
    )
    if numeric:
        value = int(numeric.group(1).replace(",", ""))
        words = _cardinal_words(value)
        if numeric.group(2):
            words = _ordinal_words(words)
        return words
    return tuple(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", normalized))


def _cardinal_words(value: int) -> tuple[str, ...]:
    if value < 0 or value >= 1_000_000_000:
        raise RuntimeError("WhisperX alignment normalization supports numbers below one billion")
    if value < 1_000:
        return _under_one_thousand_words(value)
    result: list[str] = []
    for divisor, magnitude in ((1_000_000, "million"), (1_000, "thousand")):
        group, value = divmod(value, divisor)
        if group:
            result.extend(_under_one_thousand_words(group))
            result.append(magnitude)
    if value:
        result.extend(_under_one_thousand_words(value))
    return tuple(result)


def _under_one_thousand_words(value: int) -> tuple[str, ...]:
    ones = (
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    )
    teens = (
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    result: list[str] = []
    if value >= 100:
        result.extend((ones[value // 100], "hundred"))
        value %= 100
    if value >= 20:
        result.append(tens[value // 10])
        value %= 10
    if value >= 10:
        result.append(teens[value - 10])
        value = 0
    if value:
        result.append(ones[value])
    return tuple(result or ["zero"])


def _ordinal_words(words: tuple[str, ...]) -> tuple[str, ...]:
    replacement = {
        "one": "first",
        "two": "second",
        "three": "third",
        "four": "fourth",
        "five": "fifth",
        "six": "sixth",
        "seven": "seventh",
        "eight": "eighth",
        "nine": "ninth",
        "ten": "tenth",
        "eleven": "eleventh",
        "twelve": "twelfth",
    }
    final = replacement.get(words[-1])
    if final is None:
        final = f"{words[-1][:-1]}ieth" if words[-1].endswith("y") else f"{words[-1]}th"
    return (*words[:-1], final)


def _normalized_word_intervals(
    aligned: object, transcript: str, plan: _NormalizedAlignment
) -> tuple[WordInterval, ...]:
    word_rows = _timestamped_words(aligned)
    expected = tuple(word for span in plan.spans for word in span.alignment_words)
    actual = tuple(word.strip() for word, _, _ in word_rows)
    if len(actual) != len(expected) or any(
        observed.casefold() != wanted.casefold()
        for observed, wanted in zip(actual, expected, strict=True)
    ):
        raise RuntimeError(
            "WhisperX normalized alignment words do not match the normalized transcript"
        )

    result: list[WordInterval] = []
    cursor = 0
    for span in plan.spans:
        rows = word_rows[cursor : cursor + len(span.alignment_words)]
        cursor += len(span.alignment_words)
        source_text = transcript[span.char_start : span.char_end]
        aligned_text = " ".join(span.alignment_words)
        result.append(
            WordInterval(
                word=source_text,
                start=min(start for _, start, _ in rows),
                end=max(end for _, _, end in rows),
                char_start=span.char_start,
                char_end=span.char_end,
                aligned_text=aligned_text if aligned_text != source_text else None,
            )
        )
    return tuple(result)


def _timestamped_words(aligned: object) -> tuple[tuple[str, object, object], ...]:
    if not isinstance(aligned, Mapping):
        raise RuntimeError("WhisperX alignment returned no aligned word timestamps")
    segments = aligned.get("segments")
    if not isinstance(segments, Sequence):
        raise RuntimeError("WhisperX alignment returned no aligned word timestamps")
    result: list[tuple[str, object, object]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        words = segment.get("words")
        if not isinstance(words, Sequence):
            continue
        for word_data in words:
            if not isinstance(word_data, Mapping):
                continue
            word = word_data.get("word")
            if not isinstance(word, str) or "start" not in word_data or "end" not in word_data:
                raise RuntimeError("WhisperX alignment returned a word without timestamps")
            result.append((word, word_data["start"], word_data["end"]))
    if not result:
        raise RuntimeError("WhisperX alignment returned no aligned word timestamps")
    return tuple(result)


def _transcript_span(transcript: str, word: str, cursor: int) -> tuple[int, int]:
    normalized = word.strip()
    if not normalized:
        raise RuntimeError("WhisperX alignment returned an empty word")
    char_start = transcript.find(normalized, cursor)
    if char_start < 0:
        raise RuntimeError(
            "WhisperX alignment word is not present in the original supplied transcript: "
            f"{normalized!r}"
        )
    return char_start, char_start + len(normalized)


def _offset_pairs(offset_mapping: object) -> tuple[tuple[int, int], ...]:
    values = offset_mapping.tolist() if hasattr(offset_mapping, "tolist") else offset_mapping
    if isinstance(values, Sequence) and len(values) == 1:
        values = values[0]
    if not isinstance(values, Sequence):
        raise RuntimeError("BERT tokenizer did not return token offset mappings")
    result: list[tuple[int, int]] = []
    for offset in values:
        if not isinstance(offset, Sequence) or len(offset) != 2:
            raise RuntimeError("BERT tokenizer returned an invalid token offset mapping")
        start = _character_offset(offset[0], "token start")
        end = _character_offset(offset[1], "token end")
        if end <= start:
            raise RuntimeError("BERT tokenizer returned an empty token offset mapping")
        result.append((start, end))
    return tuple(result)


def _token_interval(
    transcript: str, offset: tuple[int, int], words: Sequence[WordInterval]
) -> WordInterval:
    start, end = offset
    if end > len(transcript):
        raise RuntimeError("BERT tokenizer offset exceeds the supplied transcript")
    candidates = [
        word
        for word in words
        if word.char_start < end and start < word.char_end
    ]
    if len(candidates) != 1:
        raise RuntimeError("BERT token cannot be assigned exactly one supplied word interval")
    word = candidates[0]
    return WordInterval(
        word=transcript[start:end],
        start=word.start,
        end=word.end,
        char_start=start,
        char_end=end,
    )


def _opensmile_timestamps(index: object) -> tuple[np.ndarray, np.ndarray]:
    try:
        entries = list(index)
    except TypeError as error:
        raise RuntimeError("OpenSMILE output has no timestamp index") from error
    starts: list[float] = []
    ends: list[float] = []
    for entry in entries:
        if not isinstance(entry, tuple):
            raise RuntimeError("OpenSMILE output must index each window by start and end")
        if len(entry) == 2:
            start, end = entry
        elif len(entry) == 3:
            _, start, end = entry
        else:
            raise RuntimeError("OpenSMILE output must index each window by start and end")
        starts.append(_seconds(start))
        ends.append(_seconds(end))
    return np.asarray(starts), np.asarray(ends)


def _seconds(value: object) -> float:
    if hasattr(value, "total_seconds"):
        value = value.total_seconds()
    return _finite_time(value, "timestamp")


def _face_area(landmarks: Sequence[object]) -> float:
    if not landmarks:
        raise RuntimeError("MediaPipe returned an empty face landmark set")
    xs = np.asarray([getattr(landmark, "x", np.nan) for landmark in landmarks])
    ys = np.asarray([getattr(landmark, "y", np.nan) for landmark in landmarks])
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise RuntimeError("MediaPipe returned non-finite face landmarks")
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _face_center(landmarks: Sequence[object], coordinate: str) -> float:
    values = np.asarray([getattr(landmark, coordinate, np.nan) for landmark in landmarks])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("MediaPipe returned non-finite face landmarks")
    return float((values.min() + values.max()) / 2.0)
