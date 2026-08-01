from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .inputs import GalleryDatabase, QueryRecord, timestamps_for_frames


@dataclass(frozen=True)
class CandidateResult:
    rank: int
    distance: float
    pose: Any
    record: dict[str, Any]


@dataclass(frozen=True)
class SingleFrameResult:
    query: QueryRecord
    candidates: tuple[CandidateResult, ...]
    retrieval_result: Any | None


@dataclass(frozen=True)
class HMMWindow:
    window_index: int
    queries: tuple[QueryRecord, ...]

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(query.frame_id for query in self.queries)


@dataclass(frozen=True)
class HMMWindowAssignment:
    window_index: int
    local_position: int


@dataclass(frozen=True)
class HMMFrameResult:
    window_index: int
    local_position: int
    selected: CandidateResult


@dataclass(frozen=True)
class HMMWindowResult:
    window_index: int
    frame_ids: tuple[str, ...]
    timestamps: tuple[float, ...]
    selected: tuple[CandidateResult, ...]
    transition_costs: tuple[float, ...]


def build_hmm_window_assignments(
    queries: list[QueryRecord], window_size: int = 6
) -> tuple[list[HMMWindow], dict[str, HMMWindowAssignment]]:
    if window_size < 2:
        raise ValueError("HMM 窗口大小必须至少为 2")

    valid_runs: list[list[QueryRecord]] = []
    current_run: list[QueryRecord] = []
    for query in queries:
        if query.feature_vector is None:
            if current_run:
                valid_runs.append(current_run)
                current_run = []
            continue
        current_run.append(query)
    if current_run:
        valid_runs.append(current_run)

    windows: list[HMMWindow] = []
    assignments: dict[str, HMMWindowAssignment] = {}
    for run in valid_runs:
        if len(run) < window_size:
            continue
        windows_by_start: dict[int, HMMWindow] = {}
        for start in range(len(run) - window_size + 1):
            window = HMMWindow(
                window_index=len(windows),
                queries=tuple(run[start : start + window_size]),
            )
            windows.append(window)
            windows_by_start[start] = window
        last_start = len(run) - window_size
        for local_position, query in enumerate(run):
            start = min(local_position, last_start)
            window = windows_by_start[start]
            assignments[query.frame_id] = HMMWindowAssignment(
                window_index=window.window_index,
                local_position=local_position - start,
            )
    return windows, assignments


def run_single_frame_retrieval(
    gallery: GalleryDatabase,
    queries: list[QueryRecord],
    k: int = 200,
    search_range: int = 2,
) -> dict[str, SingleFrameResult]:
    if k < 1:
        raise ValueError("K 必须至少为 1")
    if search_range < 0:
        raise ValueError("search_range 不能为负数")
    cbir = gallery.create_cbir(search_range=search_range)
    results: dict[str, SingleFrameResult] = {}
    for query in queries:
        if query.feature_vector is None:
            results[query.frame_id] = SingleFrameResult(query, (), None)
            continue
        retrieval_result = cbir.retrieve(query.feature_vector, K=k)
        candidates = tuple(
            CandidateResult(
                rank=rank,
                distance=float(distance),
                pose=pose,
                record=gallery.records_by_pose_id[id(pose)],
            )
            for rank, (pose, distance) in enumerate(
                retrieval_result.candidates, start=1
            )
        )
        results[query.frame_id] = SingleFrameResult(
            query=query,
            candidates=candidates,
            retrieval_result=retrieval_result,
        )
    return results


def _selected_candidate(
    single_frame_result: SingleFrameResult, selected_pose: Any
) -> CandidateResult:
    for candidate in single_frame_result.candidates:
        if candidate.pose is selected_pose:
            return candidate
    raise RuntimeError(
        f"HMM 返回的位姿不属于该帧候选集: {single_frame_result.query.frame_id}"
    )


def run_hmm_diagnostics(
    module: Any,
    windows: list[HMMWindow],
    assignments: dict[str, HMMWindowAssignment],
    single_frame_results: dict[str, SingleFrameResult],
    *,
    sigma_x: float = 0.6,
    sigma_y: float = 0.6,
    sigma_z: float = 3.0,
    sigma_theta: float = 2.0,
    timestamps_by_frame: dict[str, float] | None = None,
) -> tuple[dict[str, HMMFrameResult], list[HMMWindowResult]]:
    sigmas = (sigma_x, sigma_y, sigma_z, sigma_theta)
    if any(not np.isfinite(value) or value <= 0.0 for value in sigmas):
        raise ValueError("所有 HMM sigma 参数必须是有限正数")
    hmm = module.HMMPoseEstimator(
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_z=sigma_z,
        sigma_theta=sigma_theta,
    )
    window_results: list[HMMWindowResult] = []
    by_window: dict[int, HMMWindowResult] = {}
    for window in windows:
        frame_results = [
            single_frame_results[query.frame_id] for query in window.queries
        ]
        if any(result.retrieval_result is None for result in frame_results):
            raise RuntimeError(f"HMM 窗口包含不可检索帧: {window.frame_ids}")
        if any(not result.candidates for result in frame_results):
            raise RuntimeError(f"HMM 窗口存在零候选帧: {window.frame_ids}")
        if timestamps_by_frame is None:
            timestamps = [float(index) for index in range(len(frame_results))]
        else:
            timestamps = timestamps_for_frames(window.frame_ids, timestamps_by_frame)
        selected_poses = hmm.viterbi(
            [result.retrieval_result for result in frame_results], timestamps
        )
        if len(selected_poses) != len(frame_results):
            raise RuntimeError(f"HMM 未返回完整路径: {window.frame_ids}")
        selected = tuple(
            _selected_candidate(frame_result, pose)
            for frame_result, pose in zip(frame_results, selected_poses)
        )

        forward_direction = None
        if len(frame_results) >= 2:
            first_pose = frame_results[0].candidates[0].pose
            second_pose = frame_results[1].candidates[0].pose
            movement = second_pose.surface_point - first_pose.surface_point
            movement_norm = np.linalg.norm(movement)
            if movement_norm > 1e-6:
                forward_direction = movement / movement_norm
        transition_costs = tuple(
            float(
                hmm.transition_cost(
                    selected[index - 1].pose,
                    selected[index].pose,
                    dt=timestamps[index] - timestamps[index - 1],
                    forward_direction=forward_direction if index <= 2 else None,
                )
            )
            for index in range(1, len(selected))
        )
        if not all(np.isfinite(cost) for cost in transition_costs):
            raise RuntimeError(f"HMM 路径包含非有限转移代价: {window.frame_ids}")
        window_result = HMMWindowResult(
            window_index=window.window_index,
            frame_ids=window.frame_ids,
            timestamps=tuple(timestamps),
            selected=selected,
            transition_costs=transition_costs,
        )
        window_results.append(window_result)
        by_window[window.window_index] = window_result

    frame_results: dict[str, HMMFrameResult] = {}
    for frame_id, assignment in assignments.items():
        window = by_window[assignment.window_index]
        frame_results[frame_id] = HMMFrameResult(
            window_index=assignment.window_index,
            local_position=assignment.local_position,
            selected=window.selected[assignment.local_position],
        )
    return frame_results, window_results
