"""CPU baseline for direct aortic daughters; no anatomical-name inference.

The four positional inputs and the geometry.py return contract are unchanged.
Only this module owns segmentation/tracing. Extra centreline/quality fields are
available to the viewer through result['branches']; make_prediction deliberately
exports only the challenge fields. Distances are always physical millimetres.

This is a heuristic prototype, not a validated detector. Touching enhanced veins,
poor contrast, mask errors and thick slices can cause false/missed detections.
The final dataset's minimum ostium size is not yet specified: configure it below
when published, rather than inventing an anatomical list or a scoring threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import warnings
from itertools import product
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from src.geometry import unit_direction_xyz
from src.io_utils import zyx_to_physical_xyz


@dataclass(frozen=True)
class DetectionConfig:
    """Global settings, in mm/HU; never tuned to a named case.

    min_ostium_radius_mm is an optional estimated equivalent opening radius,
    NOT the seed radius. None means no unpublished challenge size cutoff.
    spur_length_mm suppresses short skeleton artefacts when finding forks.
    """

    seed_distance_mm: float = 5.0
    max_path_length_mm: float = 10.0
    min_ostium_radius_mm: float | None = None
    intensity_low_tolerance: float = 60.0
    intensity_high_tolerance: float = 100.0
    spur_length_mm: float = 1.5
    max_cross_section_ratio: float = 2.5
    max_seed_radius_mm: float = 8.0
    duplicate_tail_length_mm: float = 2.0
    duplicate_overlap_radius_factor: float = 0.75
    duplicate_overlap_fraction: float = 0.60
    duplicate_gap_hu: float = 50.0
    duplicate_gap_fraction: float = 0.70
    # User-selected strict duplicate tolerance: only nearly identical tracks
    # may contribute to the continuous centreline-overlap rule test.
    duplicate_centreline_overlap_mm: float = 5.0
    duplicate_centreline_distance_mm: float = 0.1


def _bbox(binary: np.ndarray, margin: np.ndarray) -> tuple[slice, ...]:
    # Axis projections avoid an N-by-3 coordinate array for the full CT.
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(binary.any(axis=tuple(a for a in range(3) if a != axis)))
        bounds.append(slice(max(0, int(occupied[0] - margin[axis])),
                            min(binary.shape[axis], int(occupied[-1] + margin[axis] + 1))))
    return tuple(bounds)


def _intensity_limits(ct, aorta, spacing, config):
    interior = ndi.distance_transform_edt(aorta, sampling=spacing) > min(spacing)
    valid = interior & np.isfinite(ct)
    if np.count_nonzero(valid) < 20:
        valid = aorta & np.isfinite(ct)
    values = ct[valid]
    if not values.size:
        raise ValueError("Aorta contains no finite CT intensities")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    # Per-slice medians compensate for contrast changes along the supplied mask.
    profile = np.full(ct.shape[0], np.nan)
    for z in range(ct.shape[0]):
        samples = ct[z][valid[z]]
        if samples.size >= 8:
            profile[z] = np.median(samples)
    known = np.flatnonzero(np.isfinite(profile))
    if known.size:
        profile = np.interp(np.arange(len(profile)), known, profile[known])
    else:
        profile[:] = median
    low = profile - max(config.intensity_low_tolerance, 3 * mad)
    high = profile + max(config.intensity_high_tolerance, 4 * mad)
    return low[:, None, None], high[:, None, None]


def _graph(points, spacing):
    lookup = {tuple(p): i for i, p in enumerate(points)}
    offsets = [np.array(o) for o in product((-1, 0, 1), repeat=3) if any(o)]
    adjacency = [[] for _ in points]
    for i, p in enumerate(points):
        for offset in offsets:
            j = lookup.get(tuple(p + offset))
            if j is None or j <= i:
                continue
            # Remove redundant diagonal shortcuts across a right-angle corner.
            # Without this, 26-neighbour voxel graphs contain false tiny forks.
            axes = np.flatnonzero(offset)
            redundant = False
            if len(axes) > 1:
                for axis in axes:
                    step = np.zeros(3, dtype=int)
                    step[axis] = offset[axis]
                    if tuple(p + step) in lookup or tuple(p + offset - step) in lookup:
                        redundant = True
                        break
            if not redundant:
                length = float(np.linalg.norm(offset * spacing))
                adjacency[i].append((j, length))
                adjacency[j].append((i, length))
    return adjacency


def _skeleton_points(component, spacing):
    points = np.argwhere(skeletonize(component, method="lee"))
    if len(points) >= 2:
        return points
    # Lee thinning can erase an even-width, perfectly symmetric digital tube.
    # In that degenerate case take one distance-ridge voxel per connected
    # cross-section, with a deterministic central tie break. No tissue is added.
    axis = int(np.argmax(np.asarray(component.shape) * spacing))
    depth = ndi.distance_transform_edt(np.pad(component, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    recovered = []
    for index in range(component.shape[axis]):
        plane = np.take(component, index, axis=axis)
        plane_depth = np.take(depth, index, axis=axis)
        regions, count = ndi.label(plane)
        for region in range(1, count + 1):
            coordinates = np.argwhere(regions == region)
            values = plane_depth[tuple(coordinates.T)]
            peaks = coordinates[np.isclose(values, values.max())]
            centre = coordinates.mean(axis=0)
            chosen = peaks[np.argmin(np.sum((peaks - centre) ** 2, axis=1))]
            recovered.append(np.insert(chosen, axis, index))
    return np.asarray(recovered, dtype=int).reshape(-1, 3)


def _rooted_tree(adjacency, root):
    distance = np.full(len(adjacency), np.inf)
    parent = np.full(len(adjacency), -1, dtype=int)
    distance[root] = 0.0
    queue = [(0.0, root)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost > distance[i]:
            continue
        for j, length in adjacency[i]:
            proposed = cost + length
            if proposed < distance[j] - 1e-9:
                distance[j], parent[j] = proposed, i
                heapq.heappush(queue, (proposed, j))
    children = [[] for _ in adjacency]
    for j, i in enumerate(parent):
        if i >= 0:
            children[i].append(j)
    reach = distance.copy()
    for j in np.argsort(distance)[::-1]:
        if parent[j] >= 0:
            reach[parent[j]] = max(reach[parent[j]], reach[j])
    return distance, children, reach


def _sample(array, points, order=1):
    # ``astype(float)`` copied the entire 3D CT for every small sampling call
    # when the input was float32. Duplicate checking can make thousands of
    # calls, so that hidden copy dominated both run time and memory. SciPy can
    # sample the original numeric array directly.
    values = np.asarray(array)
    return ndi.map_coordinates(values, np.asarray(points, dtype=float).T,
                               order=order, mode="constant", cval=0, prefilter=False)


def _resample_path(path, spacing, maximum):
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0) * spacing, axis=1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-8]
    path, cumulative = path[keep], cumulative[keep]
    end = min(float(cumulative[-1]), maximum)
    # Include original corners so the resampled polyline preserves arc length.
    distances = np.unique(np.r_[np.arange(0, end, 0.25), cumulative[cumulative < end], end])
    result = np.column_stack([np.interp(distances, cumulative, path[:, a]) for a in range(3)])
    return result, distances


def _cross_section(vessel, point, tangent, spacing, max_radius):
    """Radial half-voxel boundary crossings in a plane normal to the path.

    Unlike a 3D EDT at a truncated endpoint, these rays estimate the lumen
    cross-section at the seed. Return None for an open/unbounded or flat sheet.
    """
    tangent = tangent / np.linalg.norm(tangent)
    basis = np.eye(3)[np.argmin(np.abs(tangent))]
    u = np.cross(tangent, basis)
    u /= np.linalg.norm(u)
    v = np.cross(tangent, u)
    angles = np.arange(32) * (2 * np.pi / 32)
    rays = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    step = min(0.2, float(min(spacing)) / 3)
    lengths = np.arange(0, max_radius + step, step)
    positions = point + rays[:, None, :] * lengths[None, :, None] / spacing
    samples = _sample(vessel, positions.reshape(-1, 3)).reshape(32, -1)
    outside = samples < 0.5
    if np.any(~outside.any(axis=1)) or np.any(outside[:, 0]):
        return None
    first = outside.argmax(axis=1)
    before = samples[np.arange(32), first - 1]
    after = samples[np.arange(32), first]
    radii = lengths[first - 1] + step * (before - 0.5) / np.maximum(before - after, 1e-8)
    # Opposing chords are less sensitive to half-voxel skeleton offsets.
    diameters = radii[:16] + radii[16:]
    ratio = float(np.percentile(diameters, 90) / max(np.percentile(diameters, 10), 1e-8))
    radius = float(np.sqrt(np.mean(radii ** 2)))
    return radius, ratio


def _path_positions(branch, spacing, distances):
    path = np.asarray(branch["centreline_zyx"], dtype=float)
    cumulative = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(path, axis=0) * spacing, axis=1)),
    ]
    keep = np.r_[True, np.diff(cumulative) > 1e-8]
    path, cumulative = path[keep], cumulative[keep]
    return np.column_stack(
        [np.interp(distances, cumulative, path[:, axis]) for axis in range(3)]
    )


def _dense_physical_path(branch, step_mm=0.25):
    path = np.asarray(branch["centreline_xyz_mm"], dtype=float)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-8]
    path, cumulative = path[keep], cumulative[keep]
    distances = np.unique(np.r_[np.arange(0.0, cumulative[-1], step_mm), cumulative[-1]])
    samples = np.column_stack(
        [np.interp(distances, cumulative, path[:, axis]) for axis in range(3)]
    )
    return samples, distances


def _longest_close_run(close, distances):
    longest = 0.0
    start = None
    for index, is_close in enumerate(np.r_[close, False]):
        if is_close and start is None:
            start = index
        elif not is_close and start is not None:
            longest = max(longest, float(distances[index - 1] - distances[start]))
            start = None
    return longest


def _centreline_overlap_length_mm(
    first, second, distance_mm, first_dense=None, second_dense=None
):
    """Longest symmetric physical path length whose centrelines coincide."""
    first_path, first_distance = (
        first_dense if first_dense is not None else _dense_physical_path(first)
    )
    second_path, second_distance = (
        second_dense if second_dense is not None else _dense_physical_path(second)
    )
    pairwise = np.linalg.norm(
        first_path[:, None, :] - second_path[None, :, :], axis=2
    )
    first_run = _longest_close_run(pairwise.min(axis=1) < distance_mm, first_distance)
    second_run = _longest_close_run(pairwise.min(axis=0) < distance_mm, second_distance)
    # Both paths must account for the same shared extent; this avoids treating a
    # short crossing through a densely sampled long path as prolonged overlap.
    return min(first_run, second_run)


def _intensity_gap_fraction(ct, first, second, config):
    """Fraction of paired 3D stations with a persistent low-HU interval.

    A single noisy voxel is not enough: the interval must contain at least two
    interior samples and must recur along most of the compared path. Stations
    whose centres are too close to contain an independently sampled gap are not
    counted as evidence either way.
    """
    if len(first) == 0:
        return 0.0, 0
    fractions = np.linspace(0.0, 1.0, 13)
    segments = first[:, None, :] + fractions[None, :, None] * (
        second - first
    )[:, None, :]
    profiles = _sample(ct, segments.reshape(-1, 3)).reshape(len(first), -1)
    valid = np.all(np.isfinite(profiles), axis=1)
    comparable = int(np.count_nonzero(valid))
    if not comparable:
        return 0.0, 0
    profiles = profiles[valid]
    thresholds = np.minimum(profiles[:, 0], profiles[:, -1]) - config.duplicate_gap_hu
    low = profiles[:, 2:-2] <= thresholds[:, None]
    # Require adjacent samples so isolated noise does not protect a duplicate.
    clear = int(np.count_nonzero(np.any(low[:, :-1] & low[:, 1:], axis=1)))
    return clear / comparable, comparable


def _dense_paths_can_overlap(first_dense, second_dense, distance_mm):
    """Cheap necessary test for the configured centreline-overlap rule.

    The final duplicate rule cannot pass unless at least one pair of the same
    dense samples is closer than distance_mm. Rejecting all other pairs before
    CT sampling keeps distant branches out of the expensive gap check without
    changing which pair can ultimately be merged.
    """
    first_path = first_dense[0]
    second_path = second_dense[0]
    first_min, first_max = first_path.min(axis=0), first_path.max(axis=0)
    second_min, second_max = second_path.min(axis=0), second_path.max(axis=0)
    axis_gap = np.maximum(0.0, np.maximum(first_min - second_max,
                                         second_min - first_max))
    if np.linalg.norm(axis_gap) >= distance_mm:
        return False
    squared_limit = distance_mm * distance_mm
    for start in range(0, len(first_path), 64):
        delta = (
            first_path[start:start + 64, None, :]
            - second_path[None, :, :]
        )
        if np.any(np.einsum("ijk,ijk->ij", delta, delta) < squared_limit):
            return True
    return False


def _duplicate_evidence(
    first,
    second,
    image_np,
    spacing,
    config,
    same_component=True,
    first_dense=None,
    second_dense=None,
):
    """Use a low-HU gap first, then test sustained centreline overlap."""
    common_end = min(float(first["path_length_mm"]), float(second["path_length_mm"]),
                     config.max_path_length_mm)
    tail_start = max(config.seed_distance_mm,
                     common_end - config.duplicate_tail_length_mm)
    tail_distances = np.linspace(tail_start, common_end, 7)
    first_tail = _path_positions(first, spacing, tail_distances)
    second_tail = _path_positions(second, spacing, tail_distances)
    separations = np.linalg.norm((first_tail - second_tail) * spacing, axis=1)
    combined_radius = float(first["radius_mm"] + second["radius_mm"])
    overlap_limit = config.duplicate_overlap_radius_factor * combined_radius
    overlap_fraction = float(np.mean(separations <= overlap_limit))
    gap_start = min(max(1.0, config.seed_distance_mm / 2.0), common_end)
    gap_distances = np.arange(gap_start, common_end + 1e-6, 0.75)
    first_gap = _path_positions(first, spacing, gap_distances)
    second_gap = _path_positions(second, spacing, gap_distances)
    # Convert physical separation to voxel coordinates only for the lower bound
    # on resolvable gaps; actual HU sampling remains trilinear in the 3D CT.
    physical_separation = np.linalg.norm((first_gap - second_gap) * spacing, axis=1)
    resolvable = physical_separation >= 1.5 * float(min(spacing))
    gap_fraction, gap_stations = _intensity_gap_fraction(
        image_np,
        first_gap[resolvable],
        second_gap[resolvable],
        config,
    )
    clear_intensity_gap = (
        gap_stations >= 3 and gap_fraction >= config.duplicate_gap_fraction
    )
    evidence = {
        "same_component": bool(same_component),
        "common_length_mm": common_end,
        "centreline_overlap_threshold_mm": config.duplicate_centreline_overlap_mm,
        "comparable": True,
        "distal_separation_mm": float(separations[-1]),
        "distal_overlap_fraction": overlap_fraction,
        "clear_intensity_gap": clear_intensity_gap,
        "intensity_gap_fraction": float(gap_fraction),
        "intensity_gap_stations": int(gap_stations),
    }
    if clear_intensity_gap:
        return {
            **evidence,
            "centreline_check_skipped": True,
            "centreline_overlap_mm": 0.0,
            "reason": "persistent_low_intensity_gap",
            "possible_duplicate": False,
        }

    centreline_overlap = _centreline_overlap_length_mm(
        first,
        second,
        config.duplicate_centreline_distance_mm,
        first_dense=first_dense,
        second_dense=second_dense,
    )
    overlap_duplicate = (
        centreline_overlap > config.duplicate_centreline_overlap_mm + 1e-6
    )
    return {
        **evidence,
        "centreline_check_skipped": False,
        "centreline_overlap_mm": float(centreline_overlap),
        "reason": (
            "centreline_overlap_over_threshold"
            if overlap_duplicate else "no_sustained_near_identical_overlap"
        ),
        "possible_duplicate": overlap_duplicate,
    }


def _remove_duplicate_branches(results, image_np, spacing, config):
    """Collapse candidates only for sustained near-identical centrelines.

    Any pair whose centrelines remain within the configured distance for longer
    than the configured overlap length is eligible for
    merging, even when threshold fragmentation assigned different components.
    A persistent low-HU interval preserves two distinct vessels. Distal
    proximity near 10 mm is reported but no longer triggers a merge.
    """
    if len(results) < 2:
        for branch in results:
            branch["quality"]["duplicate_check"] = {
                "status": "unique",
                "same_component_comparisons": 0,
                "pairwise_comparisons": 0,
                "distant_comparisons_skipped": 0,
                "removed_candidates": 0,
                "maximum_centreline_overlap_mm": 0.0,
            }
        return results, 0

    parent = list(range(len(results)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    comparisons = [[] for _ in results]
    skipped_distant = np.zeros(len(results), dtype=int)
    dense_paths = [_dense_physical_path(branch) for branch in results]
    duplicate_pairs = []
    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            # A pair that cannot satisfy the strict <0.1 mm centreline rule can
            # never be merged. Skip its much more expensive 3D HU profile test.
            if not _dense_paths_can_overlap(
                dense_paths[first],
                dense_paths[second],
                config.duplicate_centreline_distance_mm,
            ):
                skipped_distant[first] += 1
                skipped_distant[second] += 1
                continue
            same_component = (
                results[first]["_component_id"] == results[second]["_component_id"]
            )
            evidence = _duplicate_evidence(
                results[first], results[second], image_np, spacing, config,
                same_component=same_component,
                first_dense=dense_paths[first],
                second_dense=dense_paths[second],
            )
            comparisons[first].append(evidence)
            comparisons[second].append(evidence)
            if evidence.get("possible_duplicate", False):
                union(first, second)
                duplicate_pairs.append((first, second, evidence))

    groups = {}
    for index in range(len(results)):
        groups.setdefault(find(index), []).append(index)

    kept = []
    removed = 0
    for members in groups.values():
        representative = max(
            members,
            key=lambda index: (
                float(results[index]["path_length_mm"]),
                float(results[index]["ostium_radius_mm"]),
                -float(results[index]["quality"]["cross_section_ratio"]),
            ),
        )
        branch = results[representative]
        merged = len(members) - 1
        removed += merged
        relevant = [
            evidence for first, second, evidence in duplicate_pairs
            if first in members and second in members
        ]
        branch["quality"]["duplicate_check"] = {
            "status": "representative_after_merge" if merged else "unique",
            "same_component_comparisons": sum(
                item.get("same_component", False)
                for item in comparisons[representative]
            ),
            "pairwise_comparisons": len(comparisons[representative]),
            "distant_comparisons_skipped": int(skipped_distant[representative]),
            "removed_candidates": merged,
            "minimum_distal_separation_mm": (
                min(item["distal_separation_mm"] for item in comparisons[representative]
                    if "distal_separation_mm" in item)
                if any("distal_separation_mm" in item
                       for item in comparisons[representative])
                else None
            ),
            "maximum_centreline_overlap_mm": (
                max(item["centreline_overlap_mm"]
                    for item in comparisons[representative])
                if comparisons[representative] else 0.0
            ),
            "clear_intensity_gap_seen": any(
                item.get("clear_intensity_gap", False)
                for item in comparisons[representative]
            ),
            "merge_evidence": relevant,
        }
        kept.append(branch)
    return kept, removed


def detect_branches(
    image: sitk.Image,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    search_shell: np.ndarray,
    *,
    config: DetectionConfig | None = None,
) -> list[dict[str, Any]]:
    """Return independently wall-connected, contrast-filled tubular candidates.

    Required fields: instance_id, parent_instance_id, ostium_zyx, seed_zyx,
    ostium_xyz_mm, seed_xyz_mm, radius_mm, direction_xyz.
    Extra fields: centreline_zyx, centreline_xyz_mm, path_length_mm,
    ostium_radius_mm, quality. The existing JSON writer ignores these extras.

    The supplied shell limits centreline search; a CT halo preserves full lumen
    cross-sections near its boundary. No gap-closing operation invents a direct
    connection. A fork before 5 mm is conservatively omitted because the prompt
    does not define which downstream limb should contain that trunk's seed.
    """
    config = config or DetectionConfig()
    if (not np.isfinite(config.seed_distance_mm) or config.seed_distance_mm <= 0
            or not np.isfinite(config.max_path_length_mm)
            or config.max_path_length_mm < config.seed_distance_mm):
        raise ValueError("Require 0 < seed_distance_mm <= max_path_length_mm")
    for value in (config.intensity_low_tolerance, config.intensity_high_tolerance,
                  config.spur_length_mm, config.max_cross_section_ratio,
                  config.max_seed_radius_mm, config.duplicate_tail_length_mm,
                  config.duplicate_overlap_radius_factor,
                  config.duplicate_overlap_fraction, config.duplicate_gap_hu,
                  config.duplicate_gap_fraction,
                  config.duplicate_centreline_overlap_mm,
                  config.duplicate_centreline_distance_mm):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Detection thresholds must be finite and positive")
    if (config.duplicate_overlap_fraction > 1.0
            or config.duplicate_gap_fraction > 1.0):
        raise ValueError("Duplicate-check fractions must not exceed one")
    if config.min_ostium_radius_mm is not None and (
            not np.isfinite(config.min_ostium_radius_mm) or config.min_ostium_radius_mm < 0):
        raise ValueError("min_ostium_radius_mm must be nonnegative or None")
    if image.GetDimension() != 3 or image_np.ndim != 3:
        raise ValueError("Expected a 3D CT")
    if image_np.shape != mask_np.shape or image_np.shape != search_shell.shape:
        raise ValueError("CT, mask and search_shell must have the same shape")
    if tuple(image.GetSize()) != tuple(image_np.shape[::-1]):
        raise ValueError("SimpleITK image size and CT array differ")
    spacing = np.asarray(image.GetSpacing()[::-1], dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Image spacing must be finite and positive")
    aorta_full = np.asarray(mask_np) > 0
    if not aorta_full.any():
        raise ValueError("Aorta mask is empty")
    shell_full = np.asarray(search_shell, dtype=bool) & ~aorta_full
    if not shell_full.any():
        return []

    roi = _bbox(aorta_full | shell_full, np.ceil(8.0 / spacing).astype(int))
    origin = np.array([s.start for s in roi])
    ct = np.asarray(image_np[roi], dtype=np.float32)
    aorta, shell = aorta_full[roi], shell_full[roi]
    low, high = _intensity_limits(ct, aorta, spacing, config)
    enhanced = np.isfinite(ct) & (ct >= low) & (ct <= high)
    # Retain a small full-lumen halo for skeleton endpoint and radius estimation.
    distance_to_aorta = ndi.distance_transform_edt(~aorta, sampling=spacing)
    max_search_distance = float(distance_to_aorta[shell].max())
    candidate = enhanced & ~aorta & (distance_to_aorta <= max_search_distance + 5.0)
    face_structure = ndi.generate_binary_structure(3, 1)
    labels, _ = ndi.label(candidate, structure=face_structure)
    contact = candidate & ndi.binary_dilation(aorta, structure=face_structure)

    # Exclude artificial end faces along the mask's longest physical extent.
    parent_box = _bbox(aorta, np.zeros(3, dtype=int))
    axis = int(np.argmax([(s.stop - s.start) * spacing[i] for i, s in enumerate(parent_box)]))
    axis_grid = np.arange(aorta.shape[axis])
    end = (axis_grid <= parent_box[axis].start) | (axis_grid >= parent_box[axis].stop - 1)
    reshape = [1, 1, 1]
    reshape[axis] = len(axis_grid)
    contact &= ~end.reshape(reshape)
    component_ids = np.unique(labels[contact])
    component_ids = component_ids[component_ids != 0]
    objects = ndi.find_objects(labels)
    results = []
    early_forks = 0
    for component_id in component_ids:
        component_box = objects[component_id - 1]
        # Tight padded per-component arrays keep skeletonization CPU/memory small.
        box = tuple(slice(max(0, s.start - 1), min(ct.shape[a], s.stop + 1))
                    for a, s in enumerate(component_box))
        offset = np.array([s.start for s in box])
        component = labels[box] == component_id
        wall = contact[box] & component
        openings, count = ndi.label(wall, structure=face_structure)
        if not count:
            continue
        points = _skeleton_points(component, spacing)
        if len(points) < 2:
            continue
        adjacency = _graph(points, spacing)
        vessel = component.astype(np.float32)
        parent = aorta[box]
        # Used only for locating the actual mask/CT interface, not branch direction.
        _, nearest_parent = ndi.distance_transform_edt(~parent, sampling=spacing, return_indices=True)
        for opening_id in range(1, count + 1):
            opening = np.argwhere(openings == opening_id)
            nearest = nearest_parent[(slice(None), *opening.T)].T
            interface = (opening + nearest) / 2.0
            ostium = interface.mean(axis=0)
            # Use projected wall patch area, accounting for anisotropic voxels.
            normal = np.mean((opening - nearest) * spacing, axis=0)
            if np.linalg.norm(normal) < 1e-8:
                continue
            normal /= np.linalg.norm(normal)
            area = len(opening) * float(np.prod(spacing)) / max(float(np.sum(np.abs(normal) * spacing)), 1e-8)
            opening_radius = float(np.sqrt(area / np.pi))
            if config.min_ostium_radius_mm is not None and opening_radius < config.min_ostium_radius_mm:
                continue
            root = int(np.argmin(np.linalg.norm((points - ostium) * spacing, axis=1)))
            distance, children, reach = _rooted_tree(adjacency, root)
            indices = [root]
            current = root
            stopped_at_fork = False
            while children[current]:
                significant = [j for j in children[current]
                               if reach[j] - distance[current] >= config.spur_length_mm]
                if len(significant) > 1:
                    stopped_at_fork = True
                    break
                following = max(children[current], key=lambda j: reach[j])
                indices.append(following)
                current = following
                if distance[current] > config.max_path_length_mm + 2 * max(spacing):
                    break
            raw_path = np.vstack([ostium, points[indices]])
            path, arc = _resample_path(raw_path, spacing, config.max_path_length_mm)
            if arc[-1] < config.seed_distance_mm - 1e-6:
                if stopped_at_fork:
                    early_forks += 1
                continue
            # All traced points after the opening must remain in this component;
            # this also rejects a straight root connector that cuts across tissue.
            if np.any(_sample(vessel, path[arc >= min(spacing)]) < 0.5):
                continue
            roi_path = path + offset
            if np.any(_sample(shell.astype(np.uint8), roi_path[arc >= min(spacing)], order=0) < 0.5):
                # Clip the path at the first shell exit, never jump over an exit.
                inside = _sample(shell.astype(np.uint8), roi_path, order=0) >= 0.5
                exits = np.flatnonzero((arc >= min(spacing)) & ~inside)
                if exits.size:
                    path, arc = path[:exits[0]], arc[:exits[0]]
                if not arc.size or arc[-1] < config.seed_distance_mm - 1e-6:
                    continue
            seed = np.array([np.interp(config.seed_distance_mm, arc, path[:, a]) for a in range(3)])
            before = np.array([np.interp(config.seed_distance_mm - 1.0, arc, path[:, a]) for a in range(3)])
            after = np.array([np.interp(min(arc[-1], config.seed_distance_mm + 1.0), arc, path[:, a]) for a in range(3)])
            tangent = (after - before) * spacing
            if np.linalg.norm(tangent) < 1e-8:
                continue
            section = _cross_section(vessel, seed, tangent, spacing,
                                     max_radius=config.max_seed_radius_mm)
            if section is None:
                continue
            radius, aspect_ratio = section
            # Sheet-like sections and very broad blobs are not tubular evidence.
            if aspect_ratio > config.max_cross_section_ratio or arc[-1] < 1.5 * radius:
                continue
            global_path = path + offset + origin
            global_ostium = ostium + offset + origin
            global_seed = seed + offset + origin
            ostium_xyz = zyx_to_physical_xyz(image, global_ostium)
            seed_xyz = zyx_to_physical_xyz(image, global_seed)
            if np.linalg.norm(np.asarray(seed_xyz) - ostium_xyz) < 1e-8:
                continue
            results.append({
                "_component_id": int(component_id),
                "parent_instance_id": "aorta",
                "ostium_zyx": global_ostium.tolist(),
                "seed_zyx": global_seed.tolist(),
                "ostium_xyz_mm": ostium_xyz,
                "seed_xyz_mm": seed_xyz,
                "radius_mm": radius,
                "direction_xyz": unit_direction_xyz(ostium_xyz, seed_xyz),
                "centreline_zyx": global_path.tolist(),
                "centreline_xyz_mm": [zyx_to_physical_xyz(image, p) for p in global_path],
                "path_length_mm": float(arc[-1]),
                "ostium_radius_mm": opening_radius,
                "quality": {"cross_section_ratio": aspect_ratio,
                            "stopped_at_bifurcation": stopped_at_fork,
                            "method": "intensity_wall_connection_skeleton"},
            })
    results, duplicate_count = _remove_duplicate_branches(
        results, np.asarray(image_np, dtype=np.float32), spacing, config
    )
    results.sort(key=lambda branch: tuple(branch["ostium_zyx"]))
    for index, branch in enumerate(results, start=1):
        branch.pop("_component_id", None)
        branch["instance_id"] = f"branch_{index:03d}"
    if duplicate_count:
        warnings.warn(
            f"Removed {duplicate_count} possible duplicate branch candidate(s) "
            f"because their centrelines remained within "
            f"{config.duplicate_centreline_distance_mm:g} mm for more than "
            f"{config.duplicate_centreline_overlap_mm:g} mm.",
            RuntimeWarning, stacklevel=2,
        )
    if early_forks:
        warnings.warn(
            f"{early_forks} candidate opening(s) bifurcate before the "
            f"{config.seed_distance_mm:g} mm seed distance and were omitted; "
            "the challenge must clarify seed placement for these common trunks.",
            RuntimeWarning, stacklevel=2,
        )
    return results
