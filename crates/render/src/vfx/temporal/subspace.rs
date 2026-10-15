//! Paper-driven low-rank trajectory factorisation and motion planning.
//!
//! A supported shot is represented by one or more independent moving rank-nine runs.  Each run
//! preserves the overlapping C/E factors from Liu et al. (2011), while a temporary support gap or
//! a newly observable rank starts a bounded fresh run.  Coefficients are consequently selected by
//! source frame rather than being shared across incompatible bases.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use super::numerics::{solve, symmetric_eigen};
use super::trajectory_data::{FeatureTrajectory, MAX_FRAMES, SUBSPACE_RANK, TrajectorySet};
use super::trajectory_index::TrajectoryIndex;

const MAX_ACTIVE_TRACKS: usize = 160;
const MIN_TRACK_LENGTH: usize = 20;
const WINDOW_FRAMES: usize = 50;
const WINDOW_STRIDE: usize = 5;
/// A rank-nine temporal basis needs nine samples; the independent persistent-track support
/// requirement below applies to trajectory rows, not the number of frames.
const MIN_WINDOW_FRAMES: usize = SUBSPACE_RANK;
const MIN_COMPLETE_TRACKS: usize = SUBSPACE_RANK.div_ceil(2);
const TARGET_COMPLETE_TRACKS: usize = SUBSPACE_RANK.saturating_mul(2);
const MAX_COORDINATE: f64 = 1.0e9;
const FACTOR_RESIDUAL_LIMIT: f64 = 3.0;
const RANK_EPSILON: f64 = 1.0e-12;
const RIDGE_EPSILON: f64 = 1.0e-10;
const MAX_GAUSSIAN_RADIUS: usize = 300;
/// Each anchor row competes against that track-dimension's `span` worth of per-frame Eq9 data
/// rows, so the multiplier is applied per shared (track, dimension) row: at 100x the data weight
/// the recovered constant stays within ridge epsilon of the previous run's planned position while
/// remaining a well-posed solve rather than a hard equality.
const NO_MOTION_ANCHOR_MULTIPLIER: f64 = 100.0;

/// Moving-factorisation failure categories. The reason strings are stable cache/UI codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SubspaceError {
    Cancelled,
    InvalidInput,
    InsufficientTracks,
    NumericalFailure,
}

impl SubspaceError {
    pub(super) const fn reason(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::InvalidInput => "invalidInput",
            Self::InsufficientTracks => "insufficientPersistentTracks",
            Self::NumericalFailure => "numericalFailure",
        }
    }
}

#[derive(Clone, Debug)]
struct RunData {
    range: Range<usize>,
    /// Sparse sorted `(track id, [x/y coefficients])` rows for this independent basis.
    coefficients: Vec<(u32, [[f64; SUBSPACE_RANK]; 2])>,
}
#[derive(Clone, Debug)]
struct NoMotionAnchor {
    old_frame: usize,
    previous_basis: [f64; SUBSPACE_RANK],
    shared_tracks: Vec<u32>,
}

#[derive(Debug)]
struct RunWork {
    start: usize,
    end: usize,
    coefficients: Vec<(u32, [[f64; SUBSPACE_RANK]; 2])>,
    previous: Option<WindowState>,
}

/// Shared eigen-trajectories, per-run feature coefficients, and valid discontinuity-separated
/// ranges. `basis[t][r]` is the r-th eigen-trajectory value at frame t; a coefficient row is the
/// x or y linear combination for one feature.
#[derive(Clone, Debug)]
pub(super) struct SubspaceModel {
    pub(super) basis: Vec<Option<[f64; SUBSPACE_RANK]>>,
    pub(super) segments: Vec<Range<usize>>,
    runs: Vec<RunData>,
    index: Arc<TrajectoryIndex>,
}

impl SubspaceModel {
    /// Return the coefficient row belonging to the independent run covering `frame`.
    pub(super) fn coefficient(&self, frame: usize, track: usize) -> Option<&[[f64; SUBSPACE_RANK]; 2]> {
        let run = self
            .runs
            .binary_search_by(|run| {
                if frame < run.range.start {
                    Ordering::Greater
                } else if frame >= run.range.end {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
            .ok()
            .and_then(|position| self.runs.get(position))?;
        let track = u32::try_from(track).ok()?;
        let position = run.coefficients.binary_search_by_key(&track, |entry| entry.0).ok()?;
        run.coefficients.get(position).map(|entry| &entry.1)
    }

    /// Return frame-local active IDs from the shared CSR observation index.
    #[inline]
    pub(super) fn active_tracks(&self, frame: usize) -> &[u32] {
        self.index.at(frame)
    }
}

#[derive(Clone, Debug)]
struct WindowState {
    start: usize,
    end: usize,
    complete_tracks: Vec<usize>,
}

/// Factor all usable trajectories into moving rank-nine subspaces.
pub(super) fn factorize(data: &TrajectorySet, progress: &mut dyn FnMut(usize)) -> Result<SubspaceModel, SubspaceError> {
    let segments = validate_data(data)?;
    if data.frames == 0 || data.tracks.is_empty() {
        return Err(SubspaceError::InsufficientTracks);
    }
    if cancelled() {
        return Err(SubspaceError::Cancelled);
    }

    let mut basis = vec![None; data.frames];
    let mut runs = Vec::new();
    let mut factored_any = false;

    for segment in &segments {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        if segment.end.saturating_sub(segment.start) < MIN_WINDOW_FRAMES {
            continue;
        }

        let mut cursor = segment.start;
        let mut current: Option<RunWork> = None;
        while cursor < segment.end {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some((window_len, complete_tracks)) = choose_window(data, segment, cursor) else {
                let next_cursor = if let Some(run) = current.take() {
                    let run_end = run.end;
                    finish_run(run, run_end, &mut runs);
                    cursor.max(run_end)
                } else {
                    cursor.saturating_add(WINDOW_STRIDE)
                };
                cursor = next_cursor;
                continue;
            };
            let Some(end) = cursor.checked_add(window_len) else {
                return Err(SubspaceError::InvalidInput);
            };
            if end <= cursor || end > segment.end || complete_tracks.len() < MIN_COMPLETE_TRACKS {
                let next_cursor = if let Some(run) = current.take() {
                    let run_end = run.end;
                    finish_run(run, run_end, &mut runs);
                    cursor.max(run_end)
                } else {
                    cursor.saturating_add(WINDOW_STRIDE)
                };
                cursor = next_cursor;
                continue;
            }

            // A failed moving update means that the old basis cannot express the newly observed
            // rank. Retire it at this cursor and initialise a fresh run without discarding later
            // usable source frames.
            let can_move = current.as_ref().is_some_and(|run| run.previous.is_some() && cursor < run.end);
            if can_move {
                let move_result = {
                    let Some(run) = current.as_mut() else { return Err(SubspaceError::NumericalFailure) };
                    let Some(previous) = run.previous.as_ref() else { return Err(SubspaceError::NumericalFailure) };
                    let result = move_window(data, previous, cursor, end, &complete_tracks, &mut basis, &mut run.coefficients);
                    match result {
                        Ok(state) => match window_residual_exceeds(data, &basis, &state, &run.coefficients) {
                            Ok(true) => Err(SubspaceError::NumericalFailure),
                            Ok(false) => Ok(state),
                            Err(error) => Err(error),
                        },
                        Err(error) => Err(error),
                    }
                };
                match move_result {
                    Ok(state) => {
                        if let Some(run) = current.as_mut() {
                            run.end = run.end.max(end);
                            run.previous = Some(state);
                        }
                        progress(end.min(data.frames));
                        cursor = cursor.saturating_add(WINDOW_STRIDE);
                        continue;
                    }
                    Err(SubspaceError::Cancelled) => return Err(SubspaceError::Cancelled),
                    Err(SubspaceError::InvalidInput) => return Err(SubspaceError::InvalidInput),
                    Err(SubspaceError::InsufficientTracks | SubspaceError::NumericalFailure) => {
                        if let Some(run) = current.take() {
                            clear_basis(&mut basis, cursor.min(run.end)..run.end);
                            finish_run(run, cursor, &mut runs);
                        }
                    }
                }
            } else if let Some(run) = current.take() {
                let run_end = run.end;
                finish_run(run, run_end, &mut runs);
            }

            let mut run = RunWork { start: cursor, end: cursor, coefficients: Vec::new(), previous: None };
            match initial_window(data, cursor, end, &complete_tracks, &mut basis, &mut run.coefficients) {
                Ok(state) => {
                    run.end = end;
                    run.previous = Some(state);
                    current = Some(run);
                    factored_any = true;
                    progress(end.min(data.frames));
                }
                Err(SubspaceError::Cancelled) => return Err(SubspaceError::Cancelled),
                Err(SubspaceError::InvalidInput) => return Err(SubspaceError::InvalidInput),
                Err(SubspaceError::InsufficientTracks) => {}
                Err(SubspaceError::NumericalFailure) => {}
            }
            cursor = cursor.saturating_add(WINDOW_STRIDE);
        }
        if let Some(run) = current.take() {
            let run_end = run.end;
            finish_run(run, run_end, &mut runs);
        }
    }

    if !factored_any || runs.is_empty() {
        return Err(SubspaceError::InsufficientTracks);
    }
    runs.sort_by_key(|run| run.range.start);
    let mut retained_end = 0usize;
    runs.retain(|run| {
        if run.range.start < retained_end {
            return false;
        }
        retained_end = run.range.end;
        true
    });
    // Recovery boundaries are cut before a new run starts; retain the first range defensively if
    // malformed state ever leaves an overlap.
    let mut segments_out = Vec::with_capacity(runs.len());
    let mut previous_end = 0usize;
    for run in &runs {
        if run.range.start >= run.range.end || run.range.end > data.frames {
            continue;
        }
        if run.range.start < previous_end {
            continue;
        }
        segments_out.push(run.range.clone());
        previous_end = run.range.end;
    }
    if segments_out.is_empty() {
        return Err(SubspaceError::InsufficientTracks);
    }

    project_unassigned(data, &basis, &mut runs)?;
    reject_factorization_outliers(data, &basis, &mut runs)?;
    runs.retain(|run| run.range.start < run.range.end);
    segments_out = runs.iter().map(|run| run.range.clone()).collect();
    if segments_out.is_empty() {
        return Err(SubspaceError::InsufficientTracks);
    }

    if cancelled() {
        return Err(SubspaceError::Cancelled);
    }
    Ok(SubspaceModel { basis, segments: segments_out, runs, index: Arc::clone(&data.index) })
}

/// Smooth common eigen-trajectories and return the planned per-frame basis.
pub(super) fn planned_basis(
    data: &TrajectorySet,
    model: &SubspaceModel,
    smooth_frames: f64,
    no_motion: bool,
) -> Result<Vec<Option<[f64; SUBSPACE_RANK]>>, SubspaceError> {
    let data_segments = validate_data(data)?;
    validate_model(data, model)?;
    if !smooth_frames.is_finite() || smooth_frames < 0.0 {
        return Err(SubspaceError::InvalidInput);
    }
    if cancelled() {
        return Err(SubspaceError::Cancelled);
    }

    let mut planned = vec![None; data.frames];
    let segments = if model.segments.is_empty() { data_segments } else { model.segments.clone() };
    if no_motion {
        let mut previous_constant: Option<(Range<usize>, [f64; SUBSPACE_RANK])> = None;
        for segment in &segments {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let anchor = previous_constant
                .as_ref()
                .and_then(|(previous_segment, previous_basis)| no_motion_anchor(data, model, previous_segment, *previous_basis, segment));
            let constant = fit_constant_basis(data, model, segment, anchor.as_ref())?;
            for frame in segment.clone() {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                if frame >= data.frames {
                    return Err(SubspaceError::InvalidInput);
                }
                if model.basis.get(frame).is_some_and(|value| value.is_some()) {
                    planned[frame] = constant;
                }
            }
            if let Some(value) = constant {
                previous_constant = Some((segment.clone(), value));
            } else {
                previous_constant = None;
            }
        }
        return Ok(planned);
    }

    let radius = (smooth_frames * (10.0 / 3.0)).ceil().min(MAX_GAUSSIAN_RADIUS as f64) as usize;
    if radius == 0 {
        return Ok(model.basis.clone());
    }
    let sigma = (radius as f64 / 2.0_f64.sqrt()).max(1.0e-9);
    let radius_i = radius as isize;
    for segment in &segments {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        if segment.start >= segment.end || segment.end > data.frames {
            return Err(SubspaceError::InvalidInput);
        }
        for frame in segment.clone() {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            if model.basis.get(frame).and_then(|basis| *basis).is_none() {
                continue;
            }
            let mut sum = [0.0f64; SUBSPACE_RANK];
            let mut weight_sum = 0.0f64;
            let lo = frame.saturating_sub(radius);
            let hi = frame.saturating_add(radius).min(segment.end.saturating_sub(1));
            for neighbour in lo.max(segment.start)..=hi {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                let Some(value) = model.basis.get(neighbour).and_then(|basis| *basis) else { continue };
                let distance = neighbour as isize - frame as isize;
                if distance.abs() > radius_i {
                    continue;
                }
                let d = distance as f64;
                let weight = (-0.5 * (d / sigma) * (d / sigma)).exp();
                if !weight.is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                weight_sum += weight;
                if !weight_sum.is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                for component in 0..SUBSPACE_RANK {
                    sum[component] += weight * value[component];
                    if !sum[component].is_finite() {
                        return Err(SubspaceError::NumericalFailure);
                    }
                }
            }
            if weight_sum > f64::MIN_POSITIVE && weight_sum.is_finite() {
                let mut value = [0.0f64; SUBSPACE_RANK];
                for component in 0..SUBSPACE_RANK {
                    value[component] = sum[component] / weight_sum;
                    if !value[component].is_finite() {
                        return Err(SubspaceError::NumericalFailure);
                    }
                }
                planned[frame] = Some(value);
            }
        }
    }
    Ok(planned)
}

fn validate_data(data: &TrajectorySet) -> Result<Vec<Range<usize>>, SubspaceError> {
    if data.width == 0 || data.height == 0 || data.frames > MAX_FRAMES || data.index.frames() != data.frames {
        return Err(SubspaceError::InvalidInput);
    }
    if data.width > 100_000_000 || data.height > 100_000_000 {
        return Err(SubspaceError::InvalidInput);
    }
    if !data.saliency.is_empty() && data.saliency.len() != data.frames {
        return Err(SubspaceError::InvalidInput);
    }
    if data.saliency.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SubspaceError::InvalidInput);
    }
    if data.tracks.len() > data.frames.saturating_mul(MAX_ACTIVE_TRACKS) {
        return Err(SubspaceError::InvalidInput);
    }
    if data.segment_starts.len() > data.frames.saturating_add(1) {
        return Err(SubspaceError::InvalidInput);
    }
    if data.frames == 0 {
        if data.segment_starts.iter().any(|start| *start != 0) || !data.tracks.is_empty() {
            return Err(SubspaceError::InvalidInput);
        }
        return Ok(Vec::new());
    }
    if data.segment_starts.first().copied().is_some_and(|start| start != 0) {
        return Err(SubspaceError::InvalidInput);
    }
    for pair in data.segment_starts.windows(2) {
        if pair[0] >= pair[1] || pair[1] > data.frames {
            return Err(SubspaceError::InvalidInput);
        }
    }
    let mut starts = if data.segment_starts.is_empty() { vec![0] } else { data.segment_starts.clone() };
    if starts.last().copied() == Some(data.frames) {
        starts.pop();
    }
    let mut segments = Vec::with_capacity(starts.len());
    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(data.frames);
        if start >= end || end > data.frames {
            return Err(SubspaceError::InvalidInput);
        }
        segments.push(start..end);
    }

    let max_observations = data.frames.saturating_mul(MAX_ACTIVE_TRACKS);
    let mut observations = 0usize;
    for track in &data.tracks {
        if track.start > data.frames || track.points.iter().flatten().any(|value| !value.is_finite() || f64::from(value.abs()) > MAX_COORDINATE) {
            return Err(SubspaceError::InvalidInput);
        }
        let end = track.start.checked_add(track.points.len()).ok_or(SubspaceError::InvalidInput)?;
        if end > data.frames {
            return Err(SubspaceError::InvalidInput);
        }
        observations = observations.checked_add(track.points.len()).ok_or(SubspaceError::InvalidInput)?;
        if observations > max_observations {
            return Err(SubspaceError::InvalidInput);
        }
    }
    Ok(segments)
}

fn validate_model(data: &TrajectorySet, model: &SubspaceModel) -> Result<(), SubspaceError> {
    if model.basis.len() != data.frames || model.index.frames() != data.frames || model.runs.len() != model.segments.len() {
        return Err(SubspaceError::InvalidInput);
    }
    if model.basis.iter().flatten().flatten().any(|value| !value.is_finite()) {
        return Err(SubspaceError::InvalidInput);
    }
    let mut previous_end = 0usize;
    for (run, segment) in model.runs.iter().zip(&model.segments) {
        if segment.start >= segment.end || segment.end > data.frames || segment != &run.range || segment.start < previous_end {
            return Err(SubspaceError::InvalidInput);
        }
        previous_end = segment.end;
        let mut previous_track = None;
        for (track, coefficients) in &run.coefficients {
            let Some(index) = usize::try_from(*track).ok() else {
                return Err(SubspaceError::InvalidInput);
            };
            if index >= data.tracks.len() {
                return Err(SubspaceError::InvalidInput);
            }
            if previous_track.is_some_and(|old| *track <= old) || coefficients.iter().flatten().any(|value| !value.is_finite()) {
                return Err(SubspaceError::InvalidInput);
            }
            previous_track = Some(*track);
        }
    }
    if !model.basis.iter().any(Option::is_some) {
        return Err(SubspaceError::InsufficientTracks);
    }
    Ok(())
}

fn choose_window(data: &TrajectorySet, segment: &Range<usize>, start: usize) -> Option<(usize, Vec<usize>)> {
    let remaining = segment.end.checked_sub(start)?;
    let max_len = remaining.min(WINDOW_FRAMES);
    if max_len < MIN_WINDOW_FRAMES {
        return None;
    }
    // The CSR row gives the only tracks that can be complete at this start. Sorting their end
    // times once lets shorter candidate windows admit more tracks without rescanning history.
    let mut candidates = complete_candidates(data, segment, start);
    candidates.sort_unstable_by_key(|candidate| candidate.1);
    let mut first = candidates.len();
    let mut feasible = None;
    for length in (MIN_WINDOW_FRAMES..=max_len).rev() {
        let end = start.checked_add(length)?;
        while first > 0 && candidates.get(first - 1).is_some_and(|candidate| candidate.1 >= end) {
            first -= 1;
        }
        let complete_count = candidates.len().saturating_sub(first);
        if complete_count >= TARGET_COMPLETE_TRACKS || (feasible.is_none() && complete_count >= MIN_COMPLETE_TRACKS) {
            let mut complete: Vec<usize> = candidates.get(first..)?.iter().map(|candidate| candidate.0).collect();
            complete.sort_unstable();
            if complete_count >= TARGET_COMPLETE_TRACKS {
                return Some((length, complete));
            }
            feasible = Some((length, complete));
        }
    }
    feasible
}

fn complete_candidates(data: &TrajectorySet, segment: &Range<usize>, start: usize) -> Vec<(usize, usize)> {
    let mut candidates = Vec::with_capacity(MAX_ACTIVE_TRACKS);
    for &id in data.index.at(start) {
        if cancelled() {
            break;
        }
        let Some(index) = usize::try_from(id).ok() else { continue };
        let Some(track) = data.tracks.get(index) else { continue };
        let Some(track_end) = track.start.checked_add(track.points.len()) else { continue };
        if track.rejected
            || track.points.len() < MIN_TRACK_LENGTH
            || track.start > start
            || track.start < segment.start
            || track_end > segment.end
            || complete_track_crosses_cut(data, track)
        {
            continue;
        }
        candidates.push((index, track_end));
    }
    candidates
}

fn complete_track_crosses_cut(data: &TrajectorySet, track: &FeatureTrajectory) -> bool {
    let Some(end) = track.start.checked_add(track.points.len()) else { return true };
    data.segment_starts.iter().any(|cut| *cut > track.start && *cut < end)
}

fn initial_window(
    data: &TrajectorySet,
    start: usize,
    end: usize,
    complete_tracks: &[usize],
    basis: &mut [Option<[f64; SUBSPACE_RANK]>],
    coefficients: &mut Vec<(u32, [[f64; SUBSPACE_RANK]; 2])>,
) -> Result<WindowState, SubspaceError> {
    let matrix = assemble_matrix(data, start, end, complete_tracks)?;
    let (window_basis, window_coefficients) = truncated_svd(&matrix, complete_tracks.len(), end - start)?;
    for (offset, value) in window_basis.iter().copied().enumerate() {
        let frame = start.checked_add(offset).ok_or(SubspaceError::InvalidInput)?;
        let Some(slot) = basis.get_mut(frame) else { return Err(SubspaceError::InvalidInput) };
        *slot = Some(value);
    }
    for (row, track_index) in complete_tracks.iter().copied().enumerate() {
        let track = u32::try_from(track_index).map_err(|_| SubspaceError::InvalidInput)?;
        let Some(value) = window_coefficients.get(row).copied() else { return Err(SubspaceError::NumericalFailure) };
        set_coefficient(coefficients, track, value);
    }
    Ok(WindowState { start, end, complete_tracks: complete_tracks.to_vec() })
}

fn move_window(
    data: &TrajectorySet,
    previous: &WindowState,
    start: usize,
    end: usize,
    complete_tracks: &[usize],
    basis: &mut [Option<[f64; SUBSPACE_RANK]>],
    coefficients: &mut Vec<(u32, [[f64; SUBSPACE_RANK]; 2])>,
) -> Result<WindowState, SubspaceError> {
    let overlap_end = previous.end.min(end);
    if start >= overlap_end || start < previous.start {
        return Err(SubspaceError::NumericalFailure);
    }
    let overlap_len = overlap_end - start;
    let mut overlap_basis = Vec::with_capacity(overlap_len);
    for frame in start..overlap_end {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let Some(value) = basis.get(frame).and_then(|value| *value) else { return Err(SubspaceError::NumericalFailure) };
        overlap_basis.push(value);
    }

    let shared: Vec<usize> = previous
        .complete_tracks
        .iter()
        .copied()
        .filter(|index| complete_tracks.binary_search(index).is_ok() && coefficient_for(coefficients, *index as u32).is_some())
        .collect();
    let new_tracks: Vec<usize> = complete_tracks.iter().copied().filter(|index| shared.binary_search(index).is_err()).collect();

    // Eq. 7: project each newly visible complete trajectory onto fixed overlapping E¹.
    let gram = basis_gram(&overlap_basis);
    for track_index in new_tracks.iter().copied() {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let Some(track) = data.tracks.get(track_index) else { return Err(SubspaceError::InvalidInput) };
        let mut rows = [[0.0f64; SUBSPACE_RANK]; 2];
        for dimension in 0..2 {
            let mut rhs = [0.0f64; SUBSPACE_RANK];
            for (offset, frame) in (start..overlap_end).enumerate() {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                    return Err(SubspaceError::InvalidInput);
                };
                let value = point[dimension] as f64;
                let Some(eigen) = overlap_basis.get(offset) else { return Err(SubspaceError::NumericalFailure) };
                for component in 0..SUBSPACE_RANK {
                    rhs[component] += value * eigen[component];
                    if !rhs[component].is_finite() {
                        return Err(SubspaceError::NumericalFailure);
                    }
                }
            }
            rows[dimension] = solve_regularized(gram, rhs).ok_or(SubspaceError::NumericalFailure)?;
        }
        let track_id = u32::try_from(track_index).map_err(|_| SubspaceError::InvalidInput)?;
        set_coefficient(coefficients, track_id, rows);
    }

    let new_start = overlap_end;
    if new_start < end {
        // Eq. 8: solve new E columns using fixed C¹ and projected C² rows.
        let mut normal = [[0.0f64; SUBSPACE_RANK]; SUBSPACE_RANK];
        let mut rhs_columns = vec![[0.0f64; SUBSPACE_RANK]; end - new_start];
        for track_index in shared.iter().chain(new_tracks.iter()) {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(coefficients_for_track) = coefficient_for(coefficients, *track_index as u32).copied() else { continue };
            let Some(track) = data.tracks.get(*track_index) else { return Err(SubspaceError::InvalidInput) };
            for dimension in 0..2 {
                let row = coefficients_for_track[dimension];
                for i in 0..SUBSPACE_RANK {
                    for j in 0..SUBSPACE_RANK {
                        normal[i][j] += row[i] * row[j];
                    }
                }
                for (offset, frame) in (new_start..end).enumerate() {
                    if cancelled() {
                        return Err(SubspaceError::Cancelled);
                    }
                    let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                        return Err(SubspaceError::InvalidInput);
                    };
                    let value = point[dimension] as f64;
                    let Some(column) = rhs_columns.get_mut(offset) else { return Err(SubspaceError::NumericalFailure) };
                    for component in 0..SUBSPACE_RANK {
                        column[component] += row[component] * value;
                    }
                }
            }
        }
        if normal.iter().flatten().any(|value| !value.is_finite()) || rhs_columns.iter().flatten().any(|value| !value.is_finite()) {
            return Err(SubspaceError::NumericalFailure);
        }
        for (offset, frame) in (new_start..end).enumerate() {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(value) = solve_regularized(normal, rhs_columns[offset]) else { return Err(SubspaceError::NumericalFailure) };
            let Some(slot) = basis.get_mut(frame) else { return Err(SubspaceError::InvalidInput) };
            *slot = Some(value);
        }
    }
    Ok(WindowState { start, end, complete_tracks: complete_tracks.to_vec() })
}

fn window_residual_exceeds(
    data: &TrajectorySet,
    basis: &[Option<[f64; SUBSPACE_RANK]>],
    state: &WindowState,
    coefficients: &[(u32, [[f64; SUBSPACE_RANK]; 2])],
) -> Result<bool, SubspaceError> {
    let check_start = state.start.max(state.end.saturating_sub(WINDOW_STRIDE));
    let mut observed = 0usize;
    let mut bad = 0usize;
    for track_index in &state.complete_tracks {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let Some(coefficients_for_track) = coefficient_for(coefficients, u32::try_from(*track_index).map_err(|_| SubspaceError::InvalidInput)?) else {
            continue;
        };
        let Some(track) = data.tracks.get(*track_index) else { return Err(SubspaceError::InvalidInput) };
        let end = state.end.min(track.start.saturating_add(track.points.len()));
        for frame in check_start..end {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(eigen) = basis.get(frame).and_then(|value| *value) else { continue };
            let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                return Err(SubspaceError::InvalidInput);
            };
            let error = (dot(&coefficients_for_track[0], &eigen) - point[0] as f64).hypot(dot(&coefficients_for_track[1], &eigen) - point[1] as f64);
            if !error.is_finite() {
                return Err(SubspaceError::NumericalFailure);
            }
            observed = observed.saturating_add(1);
            if error > FACTOR_RESIDUAL_LIMIT {
                bad = bad.saturating_add(1);
            }
        }
    }
    Ok(observed > 0 && bad.saturating_mul(3) > observed)
}

fn assemble_matrix(data: &TrajectorySet, start: usize, end: usize, track_indices: &[usize]) -> Result<Vec<Vec<f64>>, SubspaceError> {
    let columns = end.checked_sub(start).ok_or(SubspaceError::InvalidInput)?;
    if columns == 0 || columns > WINDOW_FRAMES {
        return Err(SubspaceError::InvalidInput);
    }
    let mut matrix = Vec::with_capacity(track_indices.len().saturating_mul(2));
    for track_index in track_indices.iter().copied() {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let Some(track) = data.tracks.get(track_index) else { return Err(SubspaceError::InvalidInput) };
        for dimension in 0..2 {
            let mut row = Vec::with_capacity(columns);
            for frame in start..end {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                    return Err(SubspaceError::InvalidInput);
                };
                row.push(point[dimension] as f64);
            }
            matrix.push(row);
        }
    }
    Ok(matrix)
}

fn truncated_svd(
    matrix: &[Vec<f64>],
    track_count: usize,
    columns: usize,
) -> Result<(Vec<[f64; SUBSPACE_RANK]>, Vec<[[f64; SUBSPACE_RANK]; 2]>), SubspaceError> {
    if track_count == 0 || columns == 0 || columns > WINDOW_FRAMES || matrix.len() != track_count.saturating_mul(2) {
        return Err(SubspaceError::InvalidInput);
    }
    let mut gram = [[0.0f64; WINDOW_FRAMES]; WINDOW_FRAMES];
    for row in matrix {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        if row.len() != columns || row.iter().any(|value| !value.is_finite()) {
            return Err(SubspaceError::InvalidInput);
        }
        for i in 0..columns {
            for j in i..columns {
                gram[i][j] += row[i] * row[j];
                if !gram[i][j].is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                gram[j][i] = gram[i][j];
            }
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(gram).ok_or(SubspaceError::NumericalFailure)?;
    let largest = eigenvalues.first().copied().unwrap_or(0.0).max(0.0);
    let mut window_basis = vec![[0.0f64; SUBSPACE_RANK]; columns];
    let mut window_coefficients = vec![[[0.0f64; SUBSPACE_RANK]; 2]; track_count];

    for component in 0..SUBSPACE_RANK {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let eigenvalue = eigenvalues.get(component).copied().unwrap_or(0.0);
        let singular = if eigenvalue > largest * RANK_EPSILON && eigenvalue > 0.0 { eigenvalue.sqrt() } else { 0.0 };
        if !singular.is_finite() {
            return Err(SubspaceError::NumericalFailure);
        }
        for column in 0..columns {
            let vector = eigenvectors[column][component];
            let value = singular * vector;
            if !value.is_finite() {
                return Err(SubspaceError::NumericalFailure);
            }
            window_basis[column][component] = value;
        }
        if singular > 0.0 {
            for row in 0..matrix.len() {
                let projection = matrix[row].iter().enumerate().map(|(column, value)| value * eigenvectors[column][component]).sum::<f64>();
                let coefficient = projection / singular;
                if !projection.is_finite() || !coefficient.is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                window_coefficients[row / 2][row % 2][component] = coefficient;
            }
        }
    }
    Ok((window_basis, window_coefficients))
}

fn basis_gram(basis: &[[f64; SUBSPACE_RANK]]) -> [[f64; SUBSPACE_RANK]; SUBSPACE_RANK] {
    let mut gram = [[0.0f64; SUBSPACE_RANK]; SUBSPACE_RANK];
    for value in basis {
        for i in 0..SUBSPACE_RANK {
            for j in 0..SUBSPACE_RANK {
                gram[i][j] += value[i] * value[j];
            }
        }
    }
    gram
}

fn solve_regularized(mut matrix: [[f64; SUBSPACE_RANK]; SUBSPACE_RANK], rhs: [f64; SUBSPACE_RANK]) -> Option<[f64; SUBSPACE_RANK]> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) || rhs.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let scale = matrix.iter().flatten().map(|value| value.abs()).fold(0.0, f64::max);
    if !scale.is_finite() {
        return None;
    }
    let ridge = scale.max(1.0) * RIDGE_EPSILON;
    for i in 0..SUBSPACE_RANK {
        matrix[i][i] += ridge;
    }
    solve(matrix, rhs)
}

fn set_coefficient(entries: &mut Vec<(u32, [[f64; SUBSPACE_RANK]; 2])>, track: u32, value: [[f64; SUBSPACE_RANK]; 2]) {
    match entries.binary_search_by_key(&track, |entry| entry.0) {
        Ok(position) => {
            if let Some(slot) = entries.get_mut(position) {
                *slot = (track, value);
            }
        }
        Err(position) => entries.insert(position, (track, value)),
    }
}

fn coefficient_for(entries: &[(u32, [[f64; SUBSPACE_RANK]; 2])], track: u32) -> Option<&[[f64; SUBSPACE_RANK]; 2]> {
    let position = entries.binary_search_by_key(&track, |entry| entry.0).ok()?;
    entries.get(position).map(|entry| &entry.1)
}

fn clear_basis(basis: &mut [Option<[f64; SUBSPACE_RANK]>], range: Range<usize>) {
    let start = range.start.min(basis.len());
    let end = range.end.min(basis.len());
    if start < end {
        for slot in &mut basis[start..end] {
            *slot = None;
        }
    }
}

fn finish_run(run: RunWork, requested_end: usize, runs: &mut Vec<RunData>) {
    let end = requested_end.min(run.end);
    if run.start < end {
        runs.push(RunData { range: run.start..end, coefficients: run.coefficients });
    }
}

fn project_unassigned(data: &TrajectorySet, basis: &[Option<[f64; SUBSPACE_RANK]>], runs: &mut [RunData]) -> Result<(), SubspaceError> {
    for run in runs {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let mut candidate_ids = Vec::new();
        for frame in run.range.clone() {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            candidate_ids.extend_from_slice(data.index.at(frame));
        }
        candidate_ids.sort_unstable();
        candidate_ids.dedup();
        for id in candidate_ids {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(index) = usize::try_from(id).ok() else { continue };
            let Some(track) = data.tracks.get(index) else { continue };
            if track.rejected
                || track.points.len() < MIN_TRACK_LENGTH
                || complete_track_crosses_cut(data, track)
                || coefficient_for(&run.coefficients, id).is_some()
            {
                continue;
            }
            let Some(end) = track.start.checked_add(track.points.len()) else { return Err(SubspaceError::InvalidInput) };
            let lo = run.range.start.max(track.start);
            let hi = run.range.end.min(end).min(data.frames);
            let mut normal = [[0.0f64; SUBSPACE_RANK]; SUBSPACE_RANK];
            let mut rhs = [[0.0f64; SUBSPACE_RANK]; 2];
            let mut samples = 0usize;
            for frame in lo..hi {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                let Some(eigen) = basis.get(frame).and_then(|value| *value) else { continue };
                let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                    return Err(SubspaceError::InvalidInput);
                };
                samples = samples.saturating_add(1);
                for i in 0..SUBSPACE_RANK {
                    for j in 0..SUBSPACE_RANK {
                        normal[i][j] += eigen[i] * eigen[j];
                    }
                }
                for dimension in 0..2 {
                    for component in 0..SUBSPACE_RANK {
                        rhs[dimension][component] += eigen[component] * point[dimension] as f64;
                    }
                }
            }
            if samples == 0 {
                continue;
            }
            let mut rows = [[0.0f64; SUBSPACE_RANK]; 2];
            for dimension in 0..2 {
                rows[dimension] = solve_regularized(normal, rhs[dimension]).ok_or(SubspaceError::NumericalFailure)?;
            }
            set_coefficient(&mut run.coefficients, id, rows);
        }
    }
    Ok(())
}

fn reject_factorization_outliers(data: &TrajectorySet, basis: &[Option<[f64; SUBSPACE_RANK]>], runs: &mut [RunData]) -> Result<(), SubspaceError> {
    for run in runs {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let mut kept = Vec::with_capacity(run.coefficients.len());
        for (id, coefficients) in run.coefficients.drain(..) {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(index) = usize::try_from(id).ok() else { continue };
            let Some(track) = data.tracks.get(index) else { continue };
            if track.rejected || complete_track_crosses_cut(data, track) {
                continue;
            }
            let Some(end) = track.start.checked_add(track.points.len()) else { return Err(SubspaceError::InvalidInput) };
            let lo = run.range.start.max(track.start);
            let hi = run.range.end.min(end).min(data.frames);
            let mut observed = false;
            let mut bad = false;
            for frame in lo..hi {
                if cancelled() {
                    return Err(SubspaceError::Cancelled);
                }
                let Some(eigen) = basis.get(frame).and_then(|value| *value) else { continue };
                let Some(point) = track.points.get(frame.checked_sub(track.start).ok_or(SubspaceError::InvalidInput)?) else {
                    return Err(SubspaceError::InvalidInput);
                };
                let error = (dot(&coefficients[0], &eigen) - point[0] as f64).hypot(dot(&coefficients[1], &eigen) - point[1] as f64);
                if !error.is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                observed = true;
                if error > FACTOR_RESIDUAL_LIMIT {
                    bad = true;
                    break;
                }
            }
            if observed && !bad {
                kept.push((id, coefficients));
            }
        }
        run.coefficients = kept;
    }
    Ok(())
}

fn no_motion_anchor(
    data: &TrajectorySet,
    model: &SubspaceModel,
    previous_segment: &Range<usize>,
    previous_basis: [f64; SUBSPACE_RANK],
    segment: &Range<usize>,
) -> Option<NoMotionAnchor> {
    if previous_segment.end == 0 || segment.start >= data.frames || previous_segment.end > segment.start {
        return None;
    }
    // A source discontinuity is an explicit hard boundary even when the runs happen to be
    // adjacent in the retained basis.
    let first_cut = match data.segment_starts.binary_search(&previous_segment.end) {
        Ok(position) | Err(position) => position,
    };
    if data.segment_starts.get(first_cut).is_some_and(|cut| *cut <= segment.start) {
        return None;
    }
    let previous_frame = previous_segment.end.saturating_sub(1);
    let previous_ids = model.active_tracks(previous_frame);
    let current_ids = model.active_tracks(segment.start);
    let mut shared_tracks = Vec::new();
    for &id in previous_ids {
        if cancelled() || current_ids.binary_search(&id).is_err() {
            continue;
        }
        let Some(index) = usize::try_from(id).ok() else { continue };
        let Some(track) = data.tracks.get(index) else { continue };
        if track.rejected || model.coefficient(previous_frame, index).is_none() || model.coefficient(segment.start, index).is_none() {
            continue;
        }
        shared_tracks.push(id);
    }
    if shared_tracks.len() < MIN_COMPLETE_TRACKS {
        return None;
    }
    Some(NoMotionAnchor { old_frame: previous_frame, previous_basis, shared_tracks })
}

fn fit_constant_basis(
    data: &TrajectorySet,
    model: &SubspaceModel,
    segment: &Range<usize>,
    anchor: Option<&NoMotionAnchor>,
) -> Result<Option<[f64; SUBSPACE_RANK]>, SubspaceError> {
    if segment.start >= segment.end || segment.end > data.frames {
        return Err(SubspaceError::InvalidInput);
    }
    let mut normal = [[0.0f64; SUBSPACE_RANK]; SUBSPACE_RANK];
    let mut rhs = [0.0f64; SUBSPACE_RANK];
    let mut samples = 0usize;
    for frame in segment.clone() {
        if cancelled() {
            return Err(SubspaceError::Cancelled);
        }
        let Some(eigen) = model.basis.get(frame).and_then(|value| *value) else { continue };
        for &id in model.active_tracks(frame) {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(index) = usize::try_from(id).ok() else { continue };
            let Some(track) = data.tracks.get(index) else { continue };
            if track.rejected {
                continue;
            }
            let Some(coefficients) = model.coefficient(frame, index) else { continue };
            let offset = match frame.checked_sub(track.start) {
                Some(value) => value,
                None => continue,
            };
            let Some(_point) = track.points.get(offset) else { continue };
            for dimension in 0..2 {
                let row = coefficients[dimension];
                let target = dot(&row, &eigen);
                for i in 0..SUBSPACE_RANK {
                    for j in 0..SUBSPACE_RANK {
                        normal[i][j] += row[i] * row[j];
                    }
                    rhs[i] += row[i] * target;
                }
            }
            samples = samples.saturating_add(1);
        }
    }

    let mut anchor_samples = 0usize;
    if let Some(anchor) = anchor {
        let span = segment.end.saturating_sub(segment.start).max(1) as f64;
        // Per (track, dimension) anchor row: the segment data contributes one row per frame for
        // the same track-dimension, so this keeps the anchor at >=1e2 relative weight.
        let weight = span * NO_MOTION_ANCHOR_MULTIPLIER;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(SubspaceError::NumericalFailure);
        }
        for &id in &anchor.shared_tracks {
            if cancelled() {
                return Err(SubspaceError::Cancelled);
            }
            let Some(index) = usize::try_from(id).ok() else { continue };
            let Some(old_coefficients) = model.coefficient(anchor.old_frame, index) else { continue };
            let Some(new_coefficients) = model.coefficient(segment.start, index) else { continue };
            for dimension in 0..2 {
                let old_target = dot(&old_coefficients[dimension], &anchor.previous_basis);
                if !old_target.is_finite() {
                    return Err(SubspaceError::NumericalFailure);
                }
                let row = new_coefficients[dimension];
                for i in 0..SUBSPACE_RANK {
                    for j in 0..SUBSPACE_RANK {
                        normal[i][j] += weight * row[i] * row[j];
                    }
                    rhs[i] += weight * row[i] * old_target;
                }
                anchor_samples = anchor_samples.saturating_add(1);
            }
        }
    }
    if samples == 0 && anchor_samples == 0 {
        return Ok(None);
    }
    let value = solve_regularized(normal, rhs).ok_or(SubspaceError::NumericalFailure)?;
    Ok(Some(value))
}

fn dot(a: &[f64; SUBSPACE_RANK], b: &[f64; SUBSPACE_RANK]) -> f64 {
    let mut sum = 0.0;
    for i in 0..SUBSPACE_RANK {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn cancelled() -> bool {
    filmcraft_media::cancel::cancelled()
}

#[cfg(test)]
mod tests {
    use super::{factorize, planned_basis};
    use crate::vfx::temporal::trajectory_data::{FeatureTrajectory, TrajectorySet};

    fn make_tracks(_frames: usize, count: usize, start: usize, length: usize, moving_after: Option<usize>) -> Vec<FeatureTrajectory> {
        (0..count)
            .map(|index| {
                let points = (0..length)
                    .map(|offset| {
                        let frame = start + offset;
                        let t = frame as f64;
                        let i = index as f64;
                        let mut x = 15.0 + i * i * 0.07 + t * 0.18 + i * t * 0.001;
                        let mut y = 26.0 + (i * 0.71).sin() * 11.0 - t * 0.12 + i * t * 0.0007;
                        let noise = (((frame.saturating_mul(31).saturating_add(index.saturating_mul(17))) % 13) as f64 - 6.0) * 0.01;
                        x += noise;
                        y -= noise * 0.5;
                        if let Some(cut) = moving_after
                            && frame >= cut
                        {
                            let dt = (frame - cut) as f64;
                            x += dt * 1.7;
                            y -= dt * 0.9;
                        }
                        [x as f32, y as f32]
                    })
                    .collect();
                FeatureTrajectory { start, points, rejected: false }
            })
            .collect()
    }

    fn set(frames: usize, tracks: Vec<FeatureTrajectory>) -> TrajectorySet {
        TrajectorySet::new(640, 360, frames, tracks, vec![0], Vec::new()).expect("checked synthetic trajectory set")
    }

    #[test]
    fn low_rank_noisy_tracks_keep_local_coefficients_and_reconstruct() {
        let data = set(64, make_tracks(64, 24, 0, 64, None));
        let model = factorize(&data, &mut |_| {}).expect("low-rank tracks factorise");
        assert!(!model.segments.is_empty());
        let basis = model.basis[30].expect("middle frame has a basis");
        let coefficients = model.coefficient(30, 4).expect("frame-local coefficient exists");
        let x = coefficients[0].iter().zip(basis).map(|(a, b)| a * b).sum::<f64>();
        let y = coefficients[1].iter().zip(basis).map(|(a, b)| a * b).sum::<f64>();
        let point = data.tracks[4].points[30];
        assert!((x - point[0] as f64).hypot(y - point[1] as f64) < 1.0);
        let planned = planned_basis(&data, &model, 0.0, false).expect("zero smoothing preserves basis");
        assert!(planned[30].is_some());
        assert!(planned[63].is_some());
    }

    #[test]
    fn unsupported_track_gap_resumes_in_a_new_run() {
        let first = make_tracks(100, 12, 0, 35, None);
        let later = make_tracks(100, 12, 50, 50, Some(75));
        let mut tracks = first;
        tracks.extend(later);
        let data = set(100, tracks);
        let model = factorize(&data, &mut |_| {}).expect("later supported run is retained");
        assert!(model.segments.len() >= 2);
        assert!(model.coefficient(80, 12).is_some());
        assert!(model.basis[80].is_some());
    }

    #[test]
    fn newly_observed_motion_reinitialises_after_static_prefix() {
        let data = set(100, make_tracks(100, 24, 0, 100, Some(50)));
        let model = factorize(&data, &mut |_| {}).expect("moving suffix factorises");
        assert!(model.segments.len() >= 2);
        assert!(model.coefficient(90, 3).is_some());
        assert!(model.basis[90].is_some());
    }

    #[test]
    fn malformed_mutation_is_rejected_before_numeric_work() {
        let mut data = set(32, make_tracks(32, 8, 0, 32, None));
        data.tracks[0].points[2][0] = f32::NAN;
        assert!(factorize(&data, &mut |_| {}).is_err());
        assert!(TrajectorySet::new(640, 360, 32, make_tracks(32, 1, 31, 2, None), vec![0], Vec::new()).is_none());
    }

    #[test]
    fn no_motion_is_constant_per_supported_run() {
        let data = set(64, make_tracks(64, 24, 0, 64, None));
        let model = factorize(&data, &mut |_| {}).expect("tracks factorise");
        let planned = planned_basis(&data, &model, 12.0, true).expect("constant basis fit");
        for segment in &model.segments {
            let Some(first) = planned[segment.start] else { continue };
            for frame in segment.clone() {
                let Some(value) = planned[frame] else { continue };
                assert_eq!(value, first);
                assert!(value.iter().all(|component| component.is_finite()));
            }
        }
    }

    #[test]
    fn independent_dynamic_track_with_large_residual_is_excluded() {
        let mut tracks = make_tracks(80, 24, 0, 80, None);
        let outlier_points = (0..40)
            .map(|offset| {
                let frame = 30 + offset;
                [((frame * 17 % 53) as f32) * 4.0, ((frame * 29 % 47) as f32) * 3.0]
            })
            .collect();
        tracks.push(FeatureTrajectory { start: 30, points: outlier_points, rejected: false });
        let data = set(80, tracks);
        let model = factorize(&data, &mut |_| {}).expect("normal tracks factorise");
        assert!(model.coefficient(60, 24).is_none());
    }
    #[test]
    fn cancellation_stops_factorisation_before_projection() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };

        let data = set(64, make_tracks(64, 24, 0, 64, None));
        let flag = Arc::new(AtomicBool::new(true));
        let result = filmcraft_media::cancel::with_cancel(&flag, || super::factorize(&data, &mut |_| {}));
        assert!(matches!(result, Err(super::SubspaceError::Cancelled)));
        flag.store(false, Ordering::Relaxed);
    }

    #[test]
    fn window_shrinks_until_redundant_track_target_is_met() {
        let mut tracks = make_tracks(50, 5, 0, 50, None);
        tracks.extend(make_tracks(50, 20, 0, 30, None));
        let data = set(50, tracks);
        let (length, complete) = super::choose_window(&data, &(0..50), 0).expect("feasible synthetic window");
        assert_eq!(length, 30);
        assert!(complete.len() >= super::TARGET_COMPLETE_TRACKS);
    }

    #[test]
    fn no_motion_recovery_anchors_shared_scene_points() {
        let data = set(100, make_tracks(100, 24, 0, 100, Some(50)));
        let model = factorize(&data, &mut |_| {}).expect("moving suffix factorises");
        assert!(model.segments.len() >= 2);
        let boundary = model.segments[1].start;
        let previous_frame = boundary.saturating_sub(1);
        let planned = planned_basis(&data, &model, 0.0, true).expect("NoMotion basis");
        let old_basis = planned[previous_frame].expect("previous constant");
        let new_basis = planned[boundary].expect("new constant");
        let old_coefficients = model.coefficient(previous_frame, 3).expect("shared old coefficients");
        let new_coefficients = model.coefficient(boundary, 3).expect("shared new coefficients");
        let old_point = [
            old_coefficients[0].iter().zip(old_basis).map(|(a, b)| a * b).sum::<f64>(),
            old_coefficients[1].iter().zip(old_basis).map(|(a, b)| a * b).sum::<f64>(),
        ];
        let new_point = [
            new_coefficients[0].iter().zip(new_basis).map(|(a, b)| a * b).sum::<f64>(),
            new_coefficients[1].iter().zip(new_basis).map(|(a, b)| a * b).sum::<f64>(),
        ];
        assert!(old_point.iter().chain(new_point.iter()).all(|value| value.is_finite()));
        assert!((old_point[0] - new_point[0]).hypot(old_point[1] - new_point[1]) < 3.0);
    }
}
