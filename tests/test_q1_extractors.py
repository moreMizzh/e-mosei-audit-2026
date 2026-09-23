from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from e_mosei_audit.q1 import extractors
from e_mosei_audit.q1.extractors import WordInterval, text_slot_features


def _fake_nltk(cache: Path, *, has_punkt_tab: bool) -> tuple[SimpleNamespace, list[tuple[str, object]]]:
    root = cache / "nltk_data"
    resource = root / "tokenizers" / "punkt_tab" / "english.pickle"
    if has_punkt_tab:
        resource.parent.mkdir(parents=True)
        resource.write_bytes(b"local punkt")
    find_calls: list[tuple[str, object]] = []

    def find(resource_name: str, paths: object = None) -> str:
        find_calls.append((resource_name, paths))
        if resource_name == "tokenizers/punkt_tab/english.pickle" and paths == [str(root)] and resource.is_file():
            return str(resource)
        raise LookupError(resource_name)

    def download(*_: object, **__: object) -> object:
        raise AssertionError("test must not call the real NLTK download hook")

    return (
        SimpleNamespace(
            data=SimpleNamespace(path=["/user/global/nltk"], find=find), download=download
        ),
        find_calls,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"word": "", "start": 0.0, "end": 0.1, "char_start": 0, "char_end": 0}, "word"),
        ({"word": "word", "start": float("nan"), "end": 0.1, "char_start": 0, "char_end": 4}, "time"),
        ({"word": "word", "start": 0.1, "end": 0.1, "char_start": 0, "char_end": 4}, "time"),
        ({"word": "word", "start": 0.0, "end": 0.1, "char_start": -1, "char_end": 4}, "character"),
        ({"word": "word", "start": 0.0, "end": 0.1, "char_start": 4, "char_end": 3}, "character"),
    ],
)
def test_word_interval_rejects_invalid_text_time_or_character_spans(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WordInterval(**kwargs)


def test_text_slot_features_pools_tokens_by_their_physical_word_intervals() -> None:
    embeddings = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    intervals = [
        WordInterval("hello", 0.0, 0.4, 0, 5),
        WordInterval("world", 0.6, 1.0, 6, 11),
    ]
    slots = np.array([[0.0, 0.5], [0.5, 1.0]])

    values, mask = text_slot_features(embeddings, intervals, slots)

    assert values.tolist() == [[1.0, 3.0], [5.0, 7.0]]
    assert mask.tolist() == [True, True]


def test_text_slot_features_uses_overlap_weighting_and_preserves_empty_slot_mask() -> None:
    embeddings = np.array([[2.0], [10.0]], dtype=np.float32)
    intervals = [
        WordInterval("first", 0.0, 0.75, 0, 5),
        WordInterval("second", 0.5, 1.0, 6, 12),
    ]
    slots = np.array([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]])

    values, mask = text_slot_features(embeddings, intervals, slots)

    np.testing.assert_allclose(values[:, 0], [2.0, 22.0 / 3.0, 0.0])
    assert mask.tolist() == [True, True, False]


def test_text_slot_features_rejects_embedding_interval_count_mismatch() -> None:
    embeddings = np.zeros((2, 3), dtype=np.float32)
    slots = np.array([[0.0, 1.0]])

    with pytest.raises(ValueError, match="matching"):
        text_slot_features(embeddings, [WordInterval("one", 0.0, 1.0, 0, 3)], slots)


def test_text_slot_features_rejects_complex_slots_before_pooling() -> None:
    with pytest.raises(ValueError, match="slots"):
        text_slot_features(
            np.array([[1.0]], dtype=np.float32),
            [WordInterval("one", 0.0, 1.0, 0, 3)],
            np.array([[0.0 + 4.0j, 1.0 + 4.0j]]),
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0], dtype=np.float32),
        np.array([[1]], dtype=np.int64),
        np.array([[np.nan]], dtype=np.float32),
        np.array([[1.0 + 2.0j]], dtype=np.complex64),
    ],
)
def test_text_slot_features_rejects_malformed_embeddings(embeddings: np.ndarray) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        text_slot_features(
            embeddings,
            [WordInterval("one", 0.0, 1.0, 0, 3)],
            np.array([[0.0, 1.0]]),
        )


@pytest.mark.parametrize(
    ("builder_name", "arguments", "package"),
    [
        ("build_whisperx_aligner", (Path("models"),), "whisperx"),
        ("build_bert_encoder", (Path("models"),), "transformers"),
        ("build_opensmile_extractor", (), "opensmile"),
        ("build_mediapipe_extractor", (Path("models"),), "mediapipe"),
    ],
)
def test_builders_translate_missing_optional_packages_to_actionable_errors(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    arguments: tuple[Path, ...],
    package: str,
) -> None:
    original_import_module = extractors.importlib.import_module

    def fail_target_package(name: str, package_: object = None) -> object:
        if name == package:
            raise ModuleNotFoundError(f"No module named {name}", name=name)
        return original_import_module(name, package_)

    monkeypatch.setattr(extractors.importlib, "import_module", fail_target_package)

    with pytest.raises(RuntimeError, match=package):
        getattr(extractors, builder_name)(*arguments)


def test_normalize_bert_embeddings_requires_token_level_768_dimensions() -> None:
    values = extractors.normalize_bert_embeddings(np.ones((2, 768), dtype=np.float64))

    assert values.shape == (2, 768)
    assert values.dtype == np.float32
    with pytest.raises(ValueError, match="768"):
        extractors.normalize_bert_embeddings(np.ones((2, 767), dtype=np.float32))


def test_normalize_mediapipe_frame_pads_blendshapes_and_emits_56_dimensions() -> None:
    values = extractors.normalize_mediapipe_frame(
        np.arange(3, dtype=np.float32),
        face_area=0.2,
        center_x=0.5,
        center_y=0.6,
        face_presence=1.0,
    )

    assert values.shape == (56,)
    assert values.dtype == np.float32
    assert values[:5].tolist() == [0.0, 1.0, 2.0, 0.0, 0.0]
    assert values[-4:].tolist() == pytest.approx([0.2, 0.5, 0.6, 1.0])


def test_normalize_opensmile_windows_requires_exactly_25_lld_dimensions() -> None:
    values, starts, ends = extractors.normalize_opensmile_windows(
        np.ones((2, 25), dtype=np.float64),
        np.array([0.0, 0.01]),
        np.array([0.025, 0.035]),
    )

    assert values.shape == (2, 25)
    assert values.dtype == np.float32
    assert starts.tolist() == [0.0, 0.01]
    assert ends.tolist() == [0.025, 0.035]
    with pytest.raises(ValueError, match="25"):
        extractors.normalize_opensmile_windows(
            np.ones((1, 24), dtype=np.float32),
            np.array([0.0]),
            np.array([0.025]),
        )


def test_feature_normalizers_reject_values_that_overflow_float32() -> None:
    with pytest.raises(ValueError, match="finite"):
        extractors.normalize_bert_embeddings(np.full((1, 768), 1e100))
    with pytest.raises(ValueError, match="finite"):
        extractors.normalize_opensmile_windows(
            np.full((1, 25), 1e100), np.array([0.0]), np.array([0.025])
        )
    with pytest.raises(ValueError, match="finite"):
        extractors.normalize_mediapipe_frame(
            np.array([1e100]),
            face_area=0.2,
            center_x=0.5,
            center_y=0.6,
            face_presence=1.0,
        )


def test_bert_factory_maps_768_dimensional_tokens_to_supplied_word_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    asset = cache / "bert-base-uncased"
    asset.mkdir(parents=True)
    (asset / "config.json").write_text("{}")
    calls: list[dict[str, object]] = []

    class Tensor:
        def __getitem__(self, _: object) -> "Tensor":
            return self

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def numpy(self) -> np.ndarray:
            return np.arange(2 * 768, dtype=np.float32).reshape(2, 768)

    class Tokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, object]:
            assert text == "hello world"
            return {
                "input_ids": object(),
                "offset_mapping": np.array([[[0, 5], [6, 11]]]),
            }

    class Model:
        def eval(self) -> None:
            return None

        def __call__(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=Tensor())

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(_: str, **kwargs: object) -> Tokenizer:
            calls.append(dict(kwargs))
            return Tokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(_: str, **kwargs: object) -> Model:
            calls.append(dict(kwargs))
            return Model()

    class NoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    fake_transformers = SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModel=AutoModel)
    fake_torch = SimpleNamespace(no_grad=NoGrad)
    _patch_optional_packages(
        monkeypatch, {"transformers": fake_transformers, "torch": fake_torch}
    )

    encoder = extractors.build_bert_encoder(cache)
    values, intervals = encoder.encode(
        "hello world",
        [
            WordInterval("hello", 0.1, 0.4, 0, 5),
            WordInterval("world", 0.6, 0.9, 6, 11),
        ],
    )

    assert values.shape == (2, 768)
    assert [(item.word, item.start, item.end) for item in intervals] == [
        ("hello", 0.1, 0.4),
        ("world", 0.6, 0.9),
    ]
    assert all(call["local_files_only"] is True for call in calls)


def test_whisperx_factory_aligns_the_original_supplied_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    (cache / "models--facebook--wav2vec2-base-960h").mkdir(parents=True)
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    nltk, find_calls = _fake_nltk(cache, has_punkt_tab=True)
    original_nltk_path = list(nltk.data.path)
    original_download = nltk.download

    class FakeWhisperX:
        segments: list[dict[str, object]] | None = None

        def __init__(self) -> None:
            self.load_count = 0

        def load_align_model(self, **kwargs: object) -> tuple[object, object]:
            assert kwargs["model_cache_only"] is True
            self.load_count += 1
            return object(), object()

        @staticmethod
        def load_audio(path: str) -> str:
            assert path == str(wav_path)
            return "audio"

        def align(self, segments: list[dict[str, object]], *_: object, **__: object) -> dict[str, object]:
            assert nltk.data.path == [str(cache / "nltk_data")]
            assert nltk.download is not original_download
            self.segments = segments
            return {
                "segments": [
                    {
                        "words": [
                            {"word": "hello", "start": 0.1, "end": 0.4},
                            {"word": "world", "start": 0.6, "end": 0.9},
                        ]
                    }
                ]
            }

    whisperx = FakeWhisperX()
    _patch_optional_packages(monkeypatch, {"whisperx": whisperx, "nltk": nltk})

    aligner = extractors.build_whisperx_aligner(cache)
    words = aligner.align(wav_path, "hello world", 1.0)
    repeated_words = aligner.align(wav_path, "hello world", 1.0)

    assert whisperx.load_count == 1
    assert find_calls == [
        ("tokenizers/punkt_tab/english.pickle", [str(cache / "nltk_data")])
    ]
    assert nltk.data.path == original_nltk_path
    assert nltk.download is original_download
    assert whisperx.segments == [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    assert words == (
        WordInterval("hello", 0.1, 0.4, 0, 5),
        WordInterval("world", 0.6, 0.9, 6, 11),
    )
    assert repeated_words == words


def test_whisperx_factory_rejects_missing_local_punkt_tab_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "model_cache"
    (cache / "models--facebook--wav2vec2-base-960h").mkdir(parents=True)
    nltk, find_calls = _fake_nltk(cache, has_punkt_tab=False)
    download = nltk.download

    class FakeWhisperX:
        @staticmethod
        def load_align_model(**_: object) -> tuple[object, object]:
            return object(), object()

    _patch_optional_packages(monkeypatch, {"whisperx": FakeWhisperX(), "nltk": nltk})

    with pytest.raises(RuntimeError, match=r"punkt_tab.*model_cache.*downloads are disabled"):
        extractors.build_whisperx_aligner(cache)

    assert find_calls == [
        ("tokenizers/punkt_tab/english.pickle", [str(cache / "nltk_data")]),
        ("tokenizers/punkt_tab/english/", [str(cache / "nltk_data")]),
    ]
    assert nltk.download is download


def test_whisperx_factory_reports_missing_nltk_as_local_punkt_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "model_cache"
    (cache / "models--facebook--wav2vec2-base-960h").mkdir(parents=True)

    class FakeWhisperX:
        @staticmethod
        def load_align_model(**_: object) -> tuple[object, object]:
            return object(), object()

    original_import_module = extractors.importlib.import_module

    def import_without_nltk(name: str, package: object = None) -> object:
        if name == "whisperx":
            return FakeWhisperX()
        if name == "nltk":
            raise ModuleNotFoundError("No module named nltk", name="nltk")
        return original_import_module(name, package)

    monkeypatch.setattr(extractors.importlib, "import_module", import_without_nltk)

    with pytest.raises(RuntimeError, match=r"punkt_tab.*model_cache.*downloads are disabled"):
        extractors.build_whisperx_aligner(cache)


def test_whisperx_aligner_blocks_upstream_download_and_restores_nltk_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    (cache / "models--facebook--wav2vec2-base-960h").mkdir(parents=True)
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    nltk, _ = _fake_nltk(cache, has_punkt_tab=True)
    original_nltk_path = list(nltk.data.path)
    original_download = nltk.download
    download_attempts: list[str] = []

    class DownloadingWhisperX:
        @staticmethod
        def load_align_model(**_: object) -> tuple[object, object]:
            return object(), object()

        @staticmethod
        def load_audio(path: str) -> str:
            assert path == str(wav_path)
            return "audio"

        def align(self, *_: object, **__: object) -> object:
            download_attempts.append("attempted")
            return nltk.download("punkt_tab")

    _patch_optional_packages(
        monkeypatch, {"whisperx": DownloadingWhisperX(), "nltk": nltk}
    )
    aligner = extractors.build_whisperx_aligner(cache)

    with pytest.raises(RuntimeError, match=r"attempted.*punkt_tab.*downloads are disabled"):
        aligner.align(wav_path, "hello", 1.0)

    assert download_attempts == ["attempted"]
    assert nltk.data.path == original_nltk_path
    assert nltk.download is original_download


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_whisperx_factory_reports_local_alignment_model_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    cache = tmp_path / "models"
    (cache / "models--facebook--wav2vec2-base-960h").mkdir(parents=True)

    class FailingWhisperX:
        @staticmethod
        def load_align_model(**_: object) -> object:
            raise error_type("missing local alignment snapshot")

    nltk, _ = _fake_nltk(cache, has_punkt_tab=True)
    _patch_optional_packages(
        monkeypatch, {"whisperx": FailingWhisperX(), "nltk": nltk}
    )

    with pytest.raises(RuntimeError, match="WhisperX.*facebook/wav2vec2-base-960h"):
        extractors.build_whisperx_aligner(cache)


def test_opensmile_factory_requests_lld_and_returns_25_dimensional_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"fake wav")
    requested: dict[str, object] = {}

    class FakeFrame:
        index = [(0.0, 0.025), (0.01, 0.035)]

        @staticmethod
        def to_numpy() -> np.ndarray:
            return np.ones((2, 25), dtype=np.float64)

    class FakeSmile:
        def __init__(self, **kwargs: object) -> None:
            requested.update(kwargs)

        @staticmethod
        def process_file(path: str) -> FakeFrame:
            assert path == str(wav_path)
            return FakeFrame()

    feature_set = object()
    feature_level = object()
    fake_opensmile = SimpleNamespace(
        Smile=FakeSmile,
        FeatureSet=SimpleNamespace(eGeMAPSv02=feature_set),
        FeatureLevel=SimpleNamespace(LowLevelDescriptors=feature_level),
    )
    _patch_optional_packages(monkeypatch, {"opensmile": fake_opensmile})

    values, starts, ends = extractors.build_opensmile_extractor().extract(wav_path)

    assert requested == {"feature_set": feature_set, "feature_level": feature_level}
    assert values.shape == (2, 25)
    assert starts.tolist() == [0.0, 0.01]
    assert ends.tolist() == [0.025, 0.035]


def test_mediapipe_factory_returns_exactly_empty_56_dimensional_frames_without_faces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "face_landmarker.task").write_bytes(b"fake model")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_000001.png").write_bytes(b"fake png")

    class FakeLandmarker:
        @staticmethod
        def detect(_: object) -> SimpleNamespace:
            return SimpleNamespace(face_landmarks=[])

    class FaceLandmarker:
        @staticmethod
        def create_from_options(_: object) -> FakeLandmarker:
            return FakeLandmarker()

    fake_mediapipe = SimpleNamespace(
        Image=SimpleNamespace(create_from_file=lambda _: object()),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=SimpleNamespace(
                FaceLandmarkerOptions=lambda **kwargs: kwargs,
                FaceLandmarker=FaceLandmarker,
                RunningMode=SimpleNamespace(IMAGE="IMAGE"),
            ),
        ),
    )
    _patch_optional_packages(monkeypatch, {"mediapipe": fake_mediapipe})

    values, starts, ends = extractors.build_mediapipe_extractor(cache).extract(frame_dir)

    assert values.shape == (0, 56)
    assert values.dtype == np.float32
    assert starts.shape == (0,)
    assert ends.shape == (0,)


def test_mediapipe_detected_face_emits_presence_not_fabricated_detector_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "face_landmarker.task").write_bytes(b"fake model")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_000001.png").write_bytes(b"fake png")

    class FakeLandmarker:
        @staticmethod
        def detect(_: object) -> SimpleNamespace:
            return SimpleNamespace(
                face_landmarks=[
                    [SimpleNamespace(x=0.2, y=0.3), SimpleNamespace(x=0.6, y=0.7)]
                ],
                face_blendshapes=[[SimpleNamespace(score=0.25)]],
                face_detection_scores=[0.05],
            )

    class FaceLandmarker:
        @staticmethod
        def create_from_options(_: object) -> FakeLandmarker:
            return FakeLandmarker()

    fake_mediapipe = SimpleNamespace(
        Image=SimpleNamespace(create_from_file=lambda _: object()),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: kwargs,
            vision=SimpleNamespace(
                FaceLandmarkerOptions=lambda **kwargs: kwargs,
                FaceLandmarker=FaceLandmarker,
                RunningMode=SimpleNamespace(IMAGE="IMAGE"),
            ),
        ),
    )
    _patch_optional_packages(monkeypatch, {"mediapipe": fake_mediapipe})

    values, starts, ends = extractors.build_mediapipe_extractor(cache).extract(frame_dir)

    assert values.shape == (1, 56)
    assert values[0, -4:].tolist() == pytest.approx([0.16, 0.4, 0.5, 1.0])
    assert starts.tolist() == [0.0]
    assert ends.tolist() == [0.1]


@pytest.mark.parametrize(
    ("builder_name", "package", "expected_asset"),
    [
        ("build_whisperx_aligner", "whisperx", "facebook/wav2vec2-base-960h"),
        ("build_bert_encoder", "transformers", "bert-base-uncased"),
        ("build_mediapipe_extractor", "mediapipe", "face_landmarker.task"),
    ],
)
def test_model_factories_report_missing_expected_local_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    package: str,
    expected_asset: str,
) -> None:
    cache = tmp_path / "empty-model-cache"
    cache.mkdir()
    packages: dict[str, object] = {
        "whisperx": object(),
        "transformers": object(),
        "torch": object(),
        "mediapipe": object(),
    }
    _patch_optional_packages(monkeypatch, packages)

    with pytest.raises(RuntimeError, match=expected_asset):
        getattr(extractors, builder_name)(cache)


def _patch_optional_packages(
    monkeypatch: pytest.MonkeyPatch, packages: dict[str, object]
) -> None:
    original_import_module = extractors.importlib.import_module

    def import_fake_package(name: str, package: object = None) -> object:
        if name in packages:
            return packages[name]
        return original_import_module(name, package)

    monkeypatch.setattr(extractors.importlib, "import_module", import_fake_package)
