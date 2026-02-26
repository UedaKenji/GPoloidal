from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
from typing import Any, Callable, Literal, Optional

import numpy as np
import scipy.sparse as sps


JsonDict = dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_array_like(data: np.ndarray | sps.spmatrix) -> str:
    h = hashlib.sha256()
    if sps.issparse(data):
        m = data.tocsr()
        h.update(str(m.shape).encode("utf-8"))
        h.update(str(m.dtype).encode("utf-8"))
        h.update(m.data.tobytes(order="C"))
        h.update(m.indices.tobytes(order="C"))
        h.update(m.indptr.tobytes(order="C"))
        return h.hexdigest()
    arr = np.ascontiguousarray(np.asarray(data))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def stable_spec_id(data: Any, prefix: str | None = None) -> str:
    payload = _canonical_json_dumps(_to_jsonable(data)).encode("utf-8")
    digest = _sha256_bytes(payload)[:16]
    return f"{prefix}_{digest}" if prefix else digest


def _cache_key_jsonable(obj: Any) -> Any:
    """Canonicalize configs for cache lookup, excluding non-essential metadata.

    This is intentionally less strict than ``_to_jsonable``/``stable_spec_id``:
    traceability should keep full config hashes, while cache reuse should depend
    only on parameters that affect the computed artifact contents.
    """
    if isinstance(obj, FileRef):
        return {"sha256": obj.sha256}
    if isinstance(obj, CallableRef):
        return {
            "module": obj.module,
            "qualname": obj.qualname,
            "source_sha256": obj.source_sha256,
        }
    if isinstance(obj, InducingPointConfig):
        data: JsonDict = {"source": _cache_key_jsonable(obj.source)}
        if obj.stride not in (None, 1):
            data["stride"] = obj.stride
        if obj.length_sq_function is not None:
            data["length_sq_function"] = _cache_key_jsonable(obj.length_sq_function)
        return data
    if isinstance(obj, ObservationMatrixConfig):
        return {
            "method": obj.method,
            "lnum": obj.lnum,
            "vessel": _cache_key_jsonable(obj.vessel),
            "camera": _cache_key_jsonable(obj.camera),
            "raytrace": _cache_key_jsonable(obj.raytrace),
            "inducing_points": _cache_key_jsonable(obj.inducing_points),
        }
    if is_dataclass(obj):
        raw = asdict(obj)
        # ``note``/``extras`` are descriptive metadata and should not affect cache reuse.
        raw = {k: v for k, v in raw.items() if k not in {"note", "extras", "package_versions"}}
        return {k: _cache_key_jsonable(v) for k, v in raw.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _cache_key_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cache_key_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def stable_cache_key_id(data: Any, prefix: str | None = None) -> str:
    payload = _canonical_json_dumps(_cache_key_jsonable(data)).encode("utf-8")
    digest = _sha256_bytes(payload)[:16]
    return f"{prefix}_{digest}" if prefix else digest


def default_cache_root(app_name: str = "gpoloidal") -> Path:
    """Return a per-user cache directory suitable for large reusable artifacts."""
    env_path = os.environ.get("GPOLOIDAL_CACHE_DIR")
    if env_path:
        return Path(env_path).expanduser()

    if sys.platform.startswith("win"):
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / app_name / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / app_name

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / app_name
    return Path.home() / ".cache" / app_name


def default_record_root(app_name: str = "gpoloidal") -> Path:
    """Return a per-user default record directory (runs/results/manifests)."""
    env_path = os.environ.get("GPOLOIDAL_RECORD_DIR")
    if env_path:
        return Path(env_path).expanduser()
    if sys.platform.startswith("win"):
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / app_name / "records"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name / "records"
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / app_name
    return Path.home() / ".local" / "state" / app_name


def collect_package_versions(package_names: list[str]) -> JsonDict:
    versions: JsonDict = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@dataclass(frozen=True)
class FileRef:
    path: str
    sha256: str
    size_bytes: int
    mtime_utc: str
    note: str | None = None

    @staticmethod
    def from_path(path: str | Path, note: str | None = None) -> "FileRef":
        p = Path(path).expanduser().resolve()
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return FileRef(
            path=str(p),
            sha256=_sha256_file(p),
            size_bytes=st.st_size,
            mtime_utc=mtime,
            note=note,
        )


@dataclass(frozen=True)
class CallableRef:
    module: str
    qualname: str
    source_file: str | None
    source_sha256: str | None
    signature: str | None
    note: str | None = None

    @staticmethod
    def from_callable(func: Callable[..., Any], note: str | None = None) -> "CallableRef":
        module = getattr(func, "__module__", "<unknown>")
        qualname = getattr(func, "__qualname__", getattr(func, "__name__", "<unknown>"))
        try:
            source = inspect.getsource(func)
            source_sha256 = _sha256_bytes(source.encode("utf-8"))
        except (OSError, TypeError):
            source_sha256 = None
        try:
            source_file = inspect.getsourcefile(func)
        except TypeError:
            source_file = None
        try:
            signature = str(inspect.signature(func))
        except (TypeError, ValueError):
            signature = None
        return CallableRef(
            module=module,
            qualname=qualname,
            source_file=source_file,
            source_sha256=source_sha256,
            signature=signature,
            note=note,
        )


@dataclass(frozen=True)
class CameraConfig:
    kind: str
    params: JsonDict


@dataclass(frozen=True)
class RaytraceConfig:
    nreflections: int
    pass_through_first: bool = True
    ray_index_for_integral: int = 1
    note: str | None = None


@dataclass(frozen=True)
class VesselConfig:
    source: FileRef | None = None
    package_resource: str | None = None
    tag: str | None = None


@dataclass(frozen=True)
class InducingPointConfig:
    source: FileRef | None = None
    stride: int = 1
    length_sq_function: CallableRef | None = None
    params: JsonDict = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class ObservationMatrixConfig:
    method: str
    lnum: int
    vessel: VesselConfig
    camera: CameraConfig
    raytrace: RaytraceConfig
    inducing_points: InducingPointConfig
    package_versions: JsonDict = field(default_factory=dict)
    extras: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class PhantomConfig:
    kind: Literal["synthetic", "experimental"]
    name: str | None = None
    source: FileRef | None = None
    generator: CallableRef | None = None
    params: JsonDict = field(default_factory=dict)
    ensemble_index: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class NoiseConfig:
    model: str
    level: float | None = None
    level_definition: str | None = None
    profile: str | None = None
    seed: int | None = None
    params: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class TomographyConfig:
    model: str  # e.g. "logGP", "linGP"
    prior_kind: str
    length_scale_factor: float | None = None
    boundary_sigma: float | None = None
    boundary_value: float | None = None
    prior_mean: float | None = None
    normalize: bool = False
    obs_noise_level: float | None = None
    max_iters: int | None = None
    tol: float | None = None
    extras: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRecord:
    name: str
    created_at_utc: str
    observation_matrix_artifact_id: str | None
    observation_matrix_config: ObservationMatrixConfig | None
    phantom: PhantomConfig | None
    noise: NoiseConfig | None
    tomography: TomographyConfig | None
    references: JsonDict = field(default_factory=dict)
    metrics: JsonDict = field(default_factory=dict)
    outputs: JsonDict = field(default_factory=dict)
    notes: str | None = None


@dataclass(frozen=True)
class MatrixArtifactRecord:
    artifact_id: str
    kind: str
    storage_format: str
    data_path: str
    manifest_path: str
    shape: tuple[int, ...]
    dtype: str
    is_sparse: bool
    spec_hash: str
    file_sha256: str
    created_at_utc: str


@dataclass(frozen=True)
class InducingPointArtifactRecord:
    artifact_id: str
    kind: str
    storage_format: str
    data_path: str
    manifest_path: str
    keys: tuple[str, ...]
    spec_hash: str
    file_sha256: str
    created_at_utc: str


@dataclass(frozen=True)
class ResultArtifactRecord:
    artifact_id: str
    kind: str
    storage_format: str
    data_path: str
    manifest_path: str
    file_sha256: str
    created_at_utc: str
    metadata: JsonDict


class ProjectStore:
    """Filesystem-backed store for cache artifacts and traceable run records.

    Default (split) layout:
    - cache_root (user cache):
      - observation_matrices/
      - inducing_points/
      - manifests/
    - record_root (user records by default; can be set project-local explicitly):
      - results/
      - manifests/
      - runs/

    Legacy unified layout is still supported by passing ``root=...``.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        cache_root: str | Path | None = None,
        record_root: str | Path | None = None,
    ) -> None:
        if root is not None and (cache_root is not None or record_root is not None):
            raise ValueError("Specify either 'root' (legacy unified layout) or 'cache_root'/'record_root', not both.")

        if root is not None:
            unified = Path(root).expanduser()
            self.cache_root = unified
            self.record_root = unified
            self.layout_mode = "unified"
        else:
            self.cache_root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
            self.record_root = Path(record_root).expanduser() if record_root is not None else default_record_root()
            self.layout_mode = "split"

        # Keep ``root`` for convenience/backward-oriented introspection.
        self.root = self.record_root

        self.obsmat_dir = self.cache_root / "observation_matrices"
        self.indpts_dir = self.cache_root / "inducing_points"
        self.cache_manifest_dir = self.cache_root / "manifests"
        self.results_dir = self.record_root / "results"
        self.record_manifest_dir = self.record_root / "manifests"
        self.run_dir = self.record_root / "runs"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        # Cache dirs and run records are common in normal operation.
        # Record-side results/manifests are created lazily only when backend result
        # artifacts are actually used, to avoid empty directories in lightweight mode.
        self.obsmat_dir.mkdir(parents=True, exist_ok=True)
        self.indpts_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manifest_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_roots(self) -> JsonDict:
        return {
            "layout_mode": self.layout_mode,
            "cache_root": str(self.cache_root.resolve()),
            "record_root": str(self.record_root.resolve()),
        }

    def _manifest_path(self, artifact_id: str) -> Path:
        if artifact_id.startswith(("obsmat_", "indpts_")):
            return self.cache_manifest_dir / f"{artifact_id}.json"
        return self.record_manifest_dir / f"{artifact_id}.json"

    def _artifact_data_path(self, artifact_id: str, suffix: str) -> Path:
        return self.obsmat_dir / f"{artifact_id}{suffix}"

    def _inducing_points_data_path(self, artifact_id: str, suffix: str = ".npz") -> Path:
        return self.indpts_dir / f"{artifact_id}{suffix}"

    def _result_data_path(self, artifact_id: str, suffix: str) -> Path:
        return self.results_dir / f"{artifact_id}{suffix}"

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _read_json(self, path: Path) -> JsonDict:
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_matrix_artifact_id(self, config: ObservationMatrixConfig) -> str:
        return stable_cache_key_id(config, prefix="obsmat")

    def _build_inducing_points_artifact_id(self, config: InducingPointConfig) -> str:
        return stable_cache_key_id(config, prefix="indpts")

    def _matrix_record_from_manifest(self, manifest: JsonDict) -> MatrixArtifactRecord:
        return MatrixArtifactRecord(
            artifact_id=manifest["artifact_id"],
            kind=manifest["kind"],
            storage_format=manifest["storage_format"],
            data_path=manifest["data_path"],
            manifest_path=manifest["manifest_path"],
            shape=tuple(manifest["shape"]),
            dtype=manifest["dtype"],
            is_sparse=bool(manifest["is_sparse"]),
            spec_hash=manifest["spec_hash"],
            file_sha256=manifest["file_sha256"],
            created_at_utc=manifest["created_at_utc"],
        )

    def _inducing_points_record_from_manifest(self, manifest: JsonDict) -> InducingPointArtifactRecord:
        return InducingPointArtifactRecord(
            artifact_id=manifest["artifact_id"],
            kind=manifest["kind"],
            storage_format=manifest["storage_format"],
            data_path=manifest["data_path"],
            manifest_path=manifest["manifest_path"],
            keys=tuple(manifest["keys"]),
            spec_hash=manifest["spec_hash"],
            file_sha256=manifest["file_sha256"],
            created_at_utc=manifest["created_at_utc"],
        )

    def _result_record_from_manifest(self, manifest: JsonDict) -> ResultArtifactRecord:
        return ResultArtifactRecord(
            artifact_id=manifest["artifact_id"],
            kind=manifest["kind"],
            storage_format=manifest["storage_format"],
            data_path=manifest["data_path"],
            manifest_path=manifest["manifest_path"],
            file_sha256=manifest["file_sha256"],
            created_at_utc=manifest["created_at_utc"],
            metadata=dict(manifest.get("metadata", {})),
        )

    def save_observation_matrix(
        self,
        config: ObservationMatrixConfig,
        matrix: np.ndarray | sps.spmatrix,
        *,
        storage_format: Literal["npy", "npz", "sparse_npz"] | None = None,
        overwrite: bool = False,
        extra_metadata: JsonDict | None = None,
    ) -> MatrixArtifactRecord:
        artifact_id = self._build_matrix_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)

        is_sparse = sps.issparse(matrix)
        if storage_format is None:
            storage_format = "sparse_npz" if is_sparse else "npy"
        if is_sparse and storage_format != "sparse_npz":
            raise ValueError("Sparse matrix must use storage_format='sparse_npz'.")
        if (not is_sparse) and storage_format == "sparse_npz":
            raise ValueError("Dense matrix cannot use storage_format='sparse_npz'.")

        suffix = {"npy": ".npy", "npz": ".npz", "sparse_npz": ".npz"}[storage_format]
        data_path = self._artifact_data_path(artifact_id, suffix)
        if data_path.exists() and not overwrite:
            if manifest_path.exists():
                return self._matrix_record_from_manifest(self._read_json(manifest_path))
            raise FileExistsError(f"Artifact data already exists without manifest: {data_path}")

        if is_sparse:
            sps.save_npz(data_path, matrix)
            shape = tuple(int(v) for v in matrix.shape)
            dtype = str(matrix.dtype)
            nnz = int(matrix.nnz)
        else:
            arr = np.asarray(matrix)
            if storage_format == "npy":
                np.save(data_path, arr, allow_pickle=False)
            else:
                np.savez_compressed(data_path, H=arr)
            shape = tuple(int(v) for v in arr.shape)
            dtype = str(arr.dtype)
            nnz = int(np.count_nonzero(arr))

        file_sha256 = _sha256_file(data_path)
        spec_hash = stable_spec_id(config)
        cache_key_hash = stable_cache_key_id(config)
        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": "observation_matrix",
            "storage_format": storage_format,
            "data_path": str(data_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "shape": shape,
            "dtype": dtype,
            "is_sparse": is_sparse,
            "nnz": nnz,
            "spec_hash": spec_hash,
            "cache_key_hash": cache_key_hash,
            "file_sha256": file_sha256,
            "created_at_utc": _utc_now_iso(),
            "config": _to_jsonable(config),
            "extra_metadata": _to_jsonable(extra_metadata or {}),
        }
        self._write_json(manifest_path, manifest)
        return self._matrix_record_from_manifest(manifest)

    def save_inducing_points(
        self,
        config: InducingPointConfig,
        arrays: dict[str, np.ndarray],
        *,
        compress: bool = True,
        overwrite: bool = False,
        extra_metadata: JsonDict | None = None,
    ) -> InducingPointArtifactRecord:
        artifact_id = self._build_inducing_points_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)
        data_path = self._inducing_points_data_path(artifact_id, ".npz")

        if data_path.exists() and not overwrite:
            if manifest_path.exists():
                return self._inducing_points_record_from_manifest(self._read_json(manifest_path))
            raise FileExistsError(f"Artifact data already exists without manifest: {data_path}")

        arrays_np = {k: np.asarray(v) for k, v in arrays.items()}
        if compress:
            np.savez_compressed(data_path, **arrays_np)
            storage_format = "npz"
        else:
            np.savez(data_path, **arrays_np)
            storage_format = "npz"

        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": "inducing_points",
            "storage_format": storage_format,
            "data_path": str(data_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "keys": sorted(arrays_np.keys()),
            "array_meta": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in arrays_np.items()
            },
            "spec_hash": stable_spec_id(config),
            "cache_key_hash": stable_cache_key_id(config),
            "file_sha256": _sha256_file(data_path),
            "created_at_utc": _utc_now_iso(),
            "config": _to_jsonable(config),
            "extra_metadata": _to_jsonable(extra_metadata or {}),
        }
        self._write_json(manifest_path, manifest)
        return self._inducing_points_record_from_manifest(manifest)

    def import_inducing_points(
        self,
        config: InducingPointConfig,
        source_path: str | Path,
        *,
        copy: bool = True,
        extra_metadata: JsonDict | None = None,
    ) -> InducingPointArtifactRecord:
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        if src.suffix.lower() != ".npz":
            raise ValueError("Inducing-point artifact import currently expects a .npz file.")

        artifact_id = self._build_inducing_points_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)
        dst = self._inducing_points_data_path(artifact_id, ".npz")
        data_path = dst if copy else src
        if copy:
            shutil.copy2(src, dst)

        arrays = self.load_inducing_points_data_path(data_path)
        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": "inducing_points",
            "storage_format": "npz",
            "data_path": str(Path(data_path).resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "keys": sorted(arrays.keys()),
            "array_meta": {
                k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
                for k, v in arrays.items()
            },
            "spec_hash": stable_spec_id(config),
            "cache_key_hash": stable_cache_key_id(config),
            "file_sha256": _sha256_file(Path(data_path)),
            "created_at_utc": _utc_now_iso(),
            "config": _to_jsonable(config),
            "extra_metadata": _to_jsonable(extra_metadata or {}),
            "imported_from": str(src),
        }
        self._write_json(manifest_path, manifest)
        return self._inducing_points_record_from_manifest(manifest)

    def get_or_build_inducing_points(
        self,
        config: InducingPointConfig,
        builder: Callable[[], dict[str, np.ndarray]],
        *,
        compress: bool = True,
        extra_metadata: JsonDict | None = None,
    ) -> tuple[dict[str, np.ndarray], InducingPointArtifactRecord]:
        artifact_id = self._build_inducing_points_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)
        if manifest_path.exists():
            record = self._inducing_points_record_from_manifest(self._read_json(manifest_path))
            print(f"[gpoloidal cache hit] inducing_points {record.artifact_id} -> {record.data_path}")
            arrays = self.load_inducing_points(record.artifact_id)
            return arrays, record
        arrays = builder()
        record = self.save_inducing_points(
            config=config,
            arrays=arrays,
            compress=compress,
            extra_metadata=extra_metadata,
        )
        return arrays, record

    def import_observation_matrix(
        self,
        config: ObservationMatrixConfig,
        source_path: str | Path,
        *,
        storage_format: Literal["npy", "npz", "sparse_npz"] | None = None,
        copy: bool = True,
        extra_metadata: JsonDict | None = None,
    ) -> MatrixArtifactRecord:
        """Register a precomputed observation matrix file with traceable config metadata."""
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        artifact_id = self._build_matrix_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)

        if storage_format is None:
            if src.suffix.lower() == ".npy":
                storage_format = "npy"
            elif src.suffix.lower() == ".npz":
                storage_format = "npz"
            else:
                raise ValueError("Could not infer storage_format. Specify 'npy'/'npz'/'sparse_npz'.")

        suffix = {"npy": ".npy", "npz": ".npz", "sparse_npz": ".npz"}[storage_format]
        dst = self._artifact_data_path(artifact_id, suffix)
        if copy:
            shutil.copy2(src, dst)
            data_path = dst
        else:
            data_path = src

        matrix = self.load_observation_matrix_data_path(data_path, storage_format=storage_format)
        is_sparse = sps.issparse(matrix)
        if is_sparse:
            shape = tuple(int(v) for v in matrix.shape)
            dtype = str(matrix.dtype)
            nnz = int(matrix.nnz)
        else:
            arr = np.asarray(matrix)
            shape = tuple(int(v) for v in arr.shape)
            dtype = str(arr.dtype)
            nnz = int(np.count_nonzero(arr))

        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": "observation_matrix",
            "storage_format": storage_format,
            "data_path": str(Path(data_path).resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "shape": shape,
            "dtype": dtype,
            "is_sparse": is_sparse,
            "nnz": nnz,
            "spec_hash": stable_spec_id(config),
            "cache_key_hash": stable_cache_key_id(config),
            "file_sha256": _sha256_file(Path(data_path)),
            "created_at_utc": _utc_now_iso(),
            "config": _to_jsonable(config),
            "extra_metadata": _to_jsonable(extra_metadata or {}),
            "imported_from": str(src),
        }
        self._write_json(manifest_path, manifest)
        return self._matrix_record_from_manifest(manifest)

    def get_or_build_observation_matrix(
        self,
        config: ObservationMatrixConfig,
        builder: Callable[[], np.ndarray | sps.spmatrix],
        *,
        storage_format: Literal["npy", "npz", "sparse_npz"] | None = None,
        extra_metadata: JsonDict | None = None,
    ) -> tuple[np.ndarray | sps.spmatrix, MatrixArtifactRecord]:
        artifact_id = self._build_matrix_artifact_id(config)
        manifest_path = self._manifest_path(artifact_id)
        if manifest_path.exists():
            record = self._matrix_record_from_manifest(self._read_json(manifest_path))
            print(f"[gpoloidal cache hit] observation_matrix {record.artifact_id} -> {record.data_path}")
            matrix = self.load_observation_matrix(record.artifact_id)
            return matrix, record

        matrix = builder()
        record = self.save_observation_matrix(
            config=config,
            matrix=matrix,
            storage_format=storage_format,
            extra_metadata=extra_metadata,
        )
        return matrix, record

    def load_observation_matrix_data_path(
        self,
        path: str | Path,
        *,
        storage_format: Literal["npy", "npz", "sparse_npz"] | None = None,
        mmap_mode: str | None = None,
    ) -> np.ndarray | sps.spmatrix:
        p = Path(path)
        if storage_format is None:
            if p.suffix.lower() == ".npy":
                storage_format = "npy"
            elif p.suffix.lower() == ".npz":
                storage_format = "npz"
            else:
                raise ValueError("Could not infer storage_format from suffix.")

        if storage_format == "sparse_npz":
            return sps.load_npz(p)
        if storage_format == "npy":
            return np.load(p, allow_pickle=False, mmap_mode=mmap_mode)
        with np.load(p, allow_pickle=False) as data:
            if "H" in data:
                return data["H"]
            if len(data.files) != 1:
                raise ValueError(f"Expected a single-array npz or key 'H', got keys={data.files}")
            return data[data.files[0]]

    def load_observation_matrix(
        self,
        artifact_id: str,
        *,
        mmap_mode: str | None = None,
    ) -> np.ndarray | sps.spmatrix:
        manifest = self._read_json(self._manifest_path(artifact_id))
        return self.load_observation_matrix_data_path(
            manifest["data_path"],
            storage_format=manifest["storage_format"],
            mmap_mode=mmap_mode,
        )

    def load_observation_matrix_record(self, artifact_id: str) -> MatrixArtifactRecord:
        return self._matrix_record_from_manifest(self._read_json(self._manifest_path(artifact_id)))

    def load_inducing_points_data_path(self, path: str | Path) -> dict[str, np.ndarray]:
        p = Path(path)
        if p.suffix.lower() != ".npz":
            raise ValueError("Inducing-point cache loader expects .npz.")
        with np.load(p, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def load_inducing_points(self, artifact_id: str) -> dict[str, np.ndarray]:
        manifest = self._read_json(self._manifest_path(artifact_id))
        return self.load_inducing_points_data_path(manifest["data_path"])

    def load_inducing_points_record(self, artifact_id: str) -> InducingPointArtifactRecord:
        return self._inducing_points_record_from_manifest(self._read_json(self._manifest_path(artifact_id)))

    def save_result_array(
        self,
        name: str,
        array: np.ndarray | sps.spmatrix,
        *,
        storage_format: Literal["npy", "npz", "sparse_npz"] | None = None,
        extra_metadata: JsonDict | None = None,
    ) -> ResultArtifactRecord:
        content_sha256 = _sha256_array_like(array)
        artifact_id = stable_spec_id(
            {
                "name": name,
                "kind": "result_array",
                "shape": getattr(array, "shape", None),
                "content_sha256": content_sha256,
            },
            prefix="res",
        )
        manifest_path = self._manifest_path(artifact_id)
        is_sparse = sps.issparse(array)
        if storage_format is None:
            storage_format = "sparse_npz" if is_sparse else "npy"
        suffix = ".npz" if storage_format in {"npz", "sparse_npz"} else ".npy"
        data_path = self._result_data_path(artifact_id, suffix)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        if is_sparse:
            if storage_format != "sparse_npz":
                raise ValueError("Sparse result array requires storage_format='sparse_npz'.")
            sps.save_npz(data_path, array)
            shape = tuple(int(v) for v in array.shape)
            dtype = str(array.dtype)
            nnz = int(array.nnz)
        else:
            arr = np.asarray(array)
            if storage_format == "npy":
                np.save(data_path, arr, allow_pickle=False)
            elif storage_format == "npz":
                np.savez_compressed(data_path, data=arr)
            else:
                raise ValueError("Dense result array supports 'npy' or 'npz'.")
            shape = tuple(int(v) for v in arr.shape)
            dtype = str(arr.dtype)
            nnz = int(np.count_nonzero(arr))

        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": "result_array",
            "storage_format": storage_format,
            "data_path": str(data_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "file_sha256": _sha256_file(data_path),
            "created_at_utc": _utc_now_iso(),
            "metadata": _to_jsonable(
                {
                    "name": name,
                    "content_sha256": content_sha256,
                    "shape": shape,
                    "dtype": dtype,
                    "nnz": nnz,
                    **(extra_metadata or {}),
                }
            ),
        }
        self._write_json(manifest_path, manifest)
        return self._result_record_from_manifest(manifest)

    def save_result_file(
        self,
        name: str,
        source_path: str | Path,
        *,
        kind: Literal["image", "file"] = "file",
        copy: bool = True,
        extra_metadata: JsonDict | None = None,
    ) -> ResultArtifactRecord:
        src = Path(source_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)

        artifact_id = stable_spec_id(
            {
                "name": name,
                "kind": kind,
                "src_name": src.name,
                "src_sha256": _sha256_file(src),
            },
            prefix="res",
        )
        manifest_path = self._manifest_path(artifact_id)
        suffix = src.suffix or ".bin"
        dst = self._result_data_path(artifact_id, suffix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data_path = dst if copy else src
        if copy:
            shutil.copy2(src, dst)

        manifest: JsonDict = {
            "artifact_id": artifact_id,
            "kind": f"result_{kind}",
            "storage_format": src.suffix.lstrip(".").lower() or "bin",
            "data_path": str(Path(data_path).resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "file_sha256": _sha256_file(Path(data_path)),
            "created_at_utc": _utc_now_iso(),
            "metadata": _to_jsonable(
                {
                    "name": name,
                    "original_filename": src.name,
                    "copied": copy,
                    "imported_from": str(src),
                    **(extra_metadata or {}),
                }
            ),
        }
        self._write_json(manifest_path, manifest)
        return self._result_record_from_manifest(manifest)

    def save_tomography_outputs(
        self,
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        covariance: np.ndarray | sps.spmatrix | None = None,
        summary_image_path: str | Path | None = None,
        prefix: str = "tomo",
        extra_metadata: JsonDict | None = None,
    ) -> JsonDict:
        outputs: JsonDict = {}
        if mean is not None:
            rec = self.save_result_array(f"{prefix}_mean", np.asarray(mean), extra_metadata=extra_metadata)
            outputs["mean_artifact_id"] = rec.artifact_id
        if std is not None:
            rec = self.save_result_array(f"{prefix}_std", np.asarray(std), extra_metadata=extra_metadata)
            outputs["std_artifact_id"] = rec.artifact_id
        if covariance is not None:
            rec = self.save_result_array(
                f"{prefix}_covariance",
                covariance,
                storage_format="sparse_npz" if sps.issparse(covariance) else "npy",
                extra_metadata=extra_metadata,
            )
            outputs["covariance_artifact_id"] = rec.artifact_id
        if summary_image_path is not None:
            rec = self.save_result_file(
                f"{prefix}_summary_image",
                summary_image_path,
                kind="image",
                extra_metadata=extra_metadata,
            )
            outputs["summary_image_artifact_id"] = rec.artifact_id
        return outputs

    def load_result_artifact_record(self, artifact_id: str) -> ResultArtifactRecord:
        return self._result_record_from_manifest(self._read_json(self._manifest_path(artifact_id)))

    def _collect_embedded_manifests(self, payload: JsonDict) -> JsonDict:
        manifests: JsonDict = {}

        obsmat_id = payload.get("observation_matrix_artifact_id")
        if isinstance(obsmat_id, str):
            mpath = self._manifest_path(obsmat_id)
            if mpath.exists():
                manifests["observation_matrix"] = self._read_json(mpath)

        refs = payload.get("references", {})
        if isinstance(refs, dict):
            indpts_id = refs.get("inducing_points_artifact_id")
            if isinstance(indpts_id, str):
                mpath = self._manifest_path(indpts_id)
                if mpath.exists():
                    manifests["inducing_points"] = self._read_json(mpath)

        outputs = payload.get("outputs", {})
        if isinstance(outputs, dict):
            result_manifests: JsonDict = {}
            for key, value in outputs.items():
                if isinstance(value, str) and value.startswith("res_"):
                    mpath = self._manifest_path(value)
                    if mpath.exists():
                        result_manifests[key] = self._read_json(mpath)
            if result_manifests:
                manifests["results"] = result_manifests

        return manifests

    def _validate_record_traceability(self, payload: JsonDict) -> None:
        obsmat_id = payload.get("observation_matrix_artifact_id")
        obsmat_cfg = payload.get("observation_matrix_config")
        if isinstance(obsmat_id, str) and isinstance(obsmat_cfg, dict):
            mpath = self._manifest_path(obsmat_id)
            if not mpath.exists():
                raise FileNotFoundError(f"Observation-matrix manifest not found: {mpath}")
            manifest = self._read_json(mpath)
            expected = stable_spec_id(obsmat_cfg)
            if manifest.get("spec_hash") != expected:
                raise ValueError(
                    "Observation-matrix config hash does not match manifest spec hash. "
                    f"artifact_id={obsmat_id}"
                )

    def save_experiment_record(
        self,
        record: ExperimentRecord,
        *,
        run_id: str | None = None,
        strict_traceability: bool = True,
        embed_dependency_manifests: bool = True,
    ) -> str:
        payload = _to_jsonable(record)
        config_hashes: JsonDict = {}
        for key in ("observation_matrix_config", "phantom", "noise", "tomography", "references"):
            value = payload.get(key)
            if value not in (None, {}):
                config_hashes[key] = stable_spec_id(value)
        payload["traceability"] = {
            "config_hashes": config_hashes,
            "strict_traceability": strict_traceability,
            "saved_at_utc": _utc_now_iso(),
        }

        if strict_traceability:
            self._validate_record_traceability(payload)

        if embed_dependency_manifests:
            payload["traceability"]["dependency_manifests"] = self._collect_embedded_manifests(payload)

        payload["traceability"]["record_hash"] = stable_spec_id(payload)

        if run_id is None:
            run_id = stable_spec_id({"name": payload.get("name"), "record_hash": payload["traceability"]["record_hash"]}, prefix="run")
        path = self.run_dir / f"{run_id}.json"
        payload["run_id"] = run_id
        self._write_json(path, payload)
        return run_id

    def load_experiment_record(self, run_id: str) -> JsonDict:
        return self._read_json(self.run_dir / f"{run_id}.json")


__all__ = [
    "CameraConfig",
    "CallableRef",
    "collect_package_versions",
    "default_cache_root",
    "default_record_root",
    "ExperimentRecord",
    "FileRef",
    "InducingPointConfig",
    "InducingPointArtifactRecord",
    "MatrixArtifactRecord",
    "NoiseConfig",
    "ObservationMatrixConfig",
    "PhantomConfig",
    "ProjectStore",
    "RaytraceConfig",
    "ResultArtifactRecord",
    "stable_cache_key_id",
    "TomographyConfig",
    "VesselConfig",
    "stable_spec_id",
]
