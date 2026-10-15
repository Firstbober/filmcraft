use std::sync::{
    Arc, Barrier, LazyLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

use super::super::mesh::MeshFrame;
use super::super::trajectory_data::GRID_CELLS;
use super::super::{
    PLAN_CACHE_BUDGET, PLAN_CACHE_MAX_ENTRIES, PlanKey, PlanProgressSnapshot, PlanSelection, PlanSettings, StabilizerPath, build_plan, cache_entry,
    cached_plan, clear_stabilizer_cache, plan_cache, plan_retained_bytes, retain_plan,
};
use super::*;
use filmcraft_project::EffectInstance;
use parking_lot::Mutex as ParkingMutex;

struct Clip {
    key: u64,
    faulty: bool,
    missing: bool,
    flag: Arc<AtomicBool>,
    cancel_at: AtomicUsize,
    reads: Mutex<Vec<usize>>,
}
impl Clip {
    fn new(key: u64, faulty: bool) -> Self {
        Self { key, faulty, missing: false, flag: Arc::new(AtomicBool::new(false)), cancel_at: AtomicUsize::new(usize::MAX), reads: Mutex::new(Vec::new()) }
    }
}
fn offset(k: usize) -> Vec2 {
    Vec2::new((k as f64 * 1.7).sin() * 0.7, (k as f64 * 2.1).cos() * 0.5)
}
impl FxEnv for Clip {
    fn frame(&self, _: f64, _: &EffectInstance) -> Option<Image> {
        None
    }
    fn source_frame(&self, dt: f64, _: f32) -> Option<Image> {
        let k = (dt * 24.0).round() as usize;
        self.reads.lock().push(k);
        if self.missing {
            return None;
        }
        if k == self.cancel_at.load(Ordering::Relaxed) {
            self.flag.store(true, Ordering::Relaxed);
        }
        if self.faulty && k == 9 {
            return Some(Image { w: usize::MAX, h: 2, px: vec![0.0; 4] });
        }
        if self.faulty && k == 25 {
            return None;
        }
        let w = if self.faulty && k == 17 { 127 } else { 128 };
        let mut img = Image::new(w, 96);
        let d = offset(k);
        for (i, p) in img.px.chunks_exact_mut(4).enumerate() {
            let (x, y) = ((i % w) as f64 + d.x, (i / w) as f64 + d.y);
            let v = if self.faulty && k == 30 { 0.5 } else { (0.5 + 0.15 * (x * 0.21).sin() * (y * 0.17).cos() + 0.12 * ((x + y) * 0.43).sin()) as f32 };
            let v = filmcraft_color::srgb_to_linear(v);
            p.copy_from_slice(&[v, v, v, 1.0]);
        }
        Some(img)
    }
    fn track(&self, _: usize) -> Option<Image> {
        None
    }
    fn layer_to_output(&self) -> Affine {
        Affine::IDENTITY
    }
    fn clip_seconds(&self) -> f64 {
        33.0 / 24.0
    }
    fn clip_offset(&self) -> f64 {
        0.0
    }
    fn frame_rate(&self) -> f64 {
        24.0
    }
    fn media_timecode(&self) -> String {
        String::new()
    }
    fn file_name(&self) -> String {
        String::new()
    }
    fn sequence_name(&self) -> String {
        String::new()
    }
    fn sequence_size(&self) -> (u32, u32) {
        (128, 96)
    }
    fn source_size(&self) -> (u32, u32) {
        (128, 96)
    }
    fn clip_key(&self) -> u64 {
        self.key
    }
}

static PLAN_TEST_LOCK: LazyLock<ParkingMutex<()>> = LazyLock::new(|| ParkingMutex::new(()));

fn plan_test_guard() -> parking_lot::MutexGuard<'static, ()> {
    PLAN_TEST_LOCK.lock()
}

fn plan_analysis(step_count: usize) -> Arc<Analysis> {
    let path = Arc::new(StabilizerPath { w: 128, h: 96, steps: vec![Affine::IDENTITY; step_count], fps: 24.0 });
    Arc::new(Analysis {
        path,
        valid_steps: vec![true; step_count],
        trajectories: None,
        subspace_model: None,
        progress: Arc::new(ParkingMutex::new(StabilizerProgress {
            stage: StabilizerStage::Planning,
            completed_frames: 0,
            total_frames: step_count.saturating_add(1),
            subspace_frames: None,
            fallback_reason: None,
        })),
        selected_plan: ParkingMutex::new(None),
        next_plan_generation: AtomicU64::new(0),
    })
}

fn plan_settings(smooth_frames: f64) -> PlanSettings {
    PlanSettings { smooth_frames, no_motion: false, preserve_scale: true, crop_less: 0.0, action_safe: 0.0, max_scale: 2.0, adaptive: false }
}

#[test]
fn batching_preserves_motion_and_never_bridges_missing_or_malformed_frames() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for faulty in [false, true] {
        let env = Clip::new(0xFA57_0000 + faulty as u64, faulty);
        let serial = analysed_path(&env, 0.0, TrackMethod::PositionScaleRotation, false, false, false).unwrap();
        let parallel = pool.install(|| analysed_path(&env, 0.0, TrackMethod::PositionScaleRotation, false, true, false)).unwrap();
        assert_eq!(serial.path.steps, parallel.path.steps);
        assert_eq!(serial.valid_steps, parallel.valid_steps);
        // The tail is not a complete batch, and the decoder still sees contiguous ordered requests.
        assert_eq!(*env.reads.lock(), (0..33).chain(0..33).collect::<Vec<_>>());
        assert_eq!(parallel.path.steps.len(), 32);
        for (k, step) in parallel.path.steps.iter().enumerate() {
            let rejected = faulty && [8, 9, 16, 17, 24, 25, 29, 30].contains(&k);
            assert_eq!(parallel.valid_steps[k], !rejected, "step {k}");
            if rejected {
                assert_eq!(*step, Affine::IDENTITY);
            } else {
                let p = Vec2::new(64.0, 48.0);
                assert!((step.apply(p) - p - (offset(k) - offset(k + 1))).length() < 0.03, "step {k}: {step:?}");
            }
        }
    }
}

#[test]
fn cancelled_analysis_is_not_published_and_can_be_retried() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for fast in [false, true] {
        let env = Clip::new(0xFA57_0010 + fast as u64, false);
        env.cancel_at.store(3, Ordering::Relaxed);
        let first = pool.install(|| filmcraft_media::cancel::with_cancel(&env.flag, || analysed_path(&env, 0.0, TrackMethod::Position, false, fast, false)));
        assert!(first.is_none());
        env.cancel_at.store(usize::MAX, Ordering::Relaxed);
        env.flag.store(false, Ordering::Relaxed);
        env.reads.lock().clear();
        let retry =
            pool.install(|| filmcraft_media::cancel::with_cancel(&env.flag, || analysed_path(&env, 0.0, TrackMethod::Position, false, fast, false))).unwrap();
        assert_eq!(retry.valid_steps, vec![true; 32]);
        assert_eq!(*env.reads.lock(), (0..33).collect::<Vec<_>>());
    }
}

#[test]
fn progress_reports_actual_completion_and_cancellation_independently_of_cache_eviction() {
    let env = Clip::new(0xFA57_0020, false);
    let slot = Slot::new(33);
    assert_eq!(
        *slot.progress.lock(),
        StabilizerProgress { stage: StabilizerStage::Analyzing, completed_frames: 0, total_frames: 33, subspace_frames: None, fallback_reason: None }
    );
    assert!(slot.get_or_analyse(&env, 0.0, TrackMethod::Position, false, true, false).is_some());
    assert_eq!(
        *slot.progress.lock(),
        StabilizerProgress { stage: StabilizerStage::Complete, completed_frames: 33, total_frames: 33, subspace_frames: None, fallback_reason: None }
    );

    env.cancel_at.store(3, Ordering::Relaxed);
    let cancelled = Slot::new(33);
    assert!(filmcraft_media::cancel::with_cancel(&env.flag, || cancelled.get_or_analyse(&env, 0.0, TrackMethod::Position, false, true, false)).is_none());
    let state = *cancelled.progress.lock();
    assert_eq!(state.stage, StabilizerStage::Cancelled);
    assert!(state.completed_frames > 0 && state.completed_frames < state.total_frames);
}

#[test]
fn missing_source_frames_finish_as_failed_instead_of_remaining_in_progress() {
    let mut env = Clip::new(0xFA57_0030, false);
    env.missing = true;
    let slot = Slot::new(33);
    assert!(slot.get_or_analyse(&env, 0.0, TrackMethod::Position, false, true, false).is_none());
    assert_eq!(
        *slot.progress.lock(),
        StabilizerProgress { stage: StabilizerStage::Failed, completed_frames: 33, total_frames: 33, subspace_frames: None, fallback_reason: None }
    );
}

#[test]
fn plan_progress_generation_rejects_stale_concurrent_updates() {
    let _guard = plan_test_guard();
    let analysis = plan_analysis(2);
    let selection_a =
        PlanSelection { key: PlanKey { path: 1, smooth: 1, crop: 0, safe: 0, maximum: 0, flags: 0 }, generation: analysis.next_plan_generation() };
    let selection_b =
        PlanSelection { key: PlanKey { path: 2, smooth: 2, crop: 0, safe: 0, maximum: 0, flags: 0 }, generation: analysis.next_plan_generation() };
    analysis.select_plan(selection_a);
    analysis.publish_plan_snapshot(
        selection_a,
        PlanProgressSnapshot { stage: StabilizerStage::Planning, completed_frames: 1, subspace_frames: None, fallback_reason: None },
    );
    let ready = Arc::new(Barrier::new(2));
    let stale_analysis = analysis.clone();
    let stale_ready = ready.clone();
    let stale = std::thread::spawn(move || {
        stale_ready.wait();
        stale_analysis.publish_plan_snapshot(
            selection_a,
            PlanProgressSnapshot { stage: StabilizerStage::Cancelled, completed_frames: 0, subspace_frames: None, fallback_reason: Some("stale") },
        );
    });
    analysis.select_plan(selection_b);
    ready.wait();
    stale.join().expect("stale publisher");
    analysis.publish_plan_snapshot(
        selection_b,
        PlanProgressSnapshot { stage: StabilizerStage::Complete, completed_frames: 3, subspace_frames: Some(2), fallback_reason: None },
    );
    assert_eq!(
        *analysis.progress.lock(),
        StabilizerProgress { stage: StabilizerStage::Complete, completed_frames: 3, total_frames: 3, subspace_frames: Some(2), fallback_reason: None }
    );
}

#[test]
fn cancelled_plan_does_not_publish_partial_and_cached_a_hit_republishes() {
    let _guard = plan_test_guard();
    clear_stabilizer_cache();
    let analysis = plan_analysis(4);
    let settings_a = plan_settings(4.0);
    let settings_b = plan_settings(8.0);
    let not_cancelled = Arc::new(AtomicBool::new(false));
    let first_a = filmcraft_media::cancel::with_cancel(&not_cancelled, || cached_plan(&analysis, &settings_a)).expect("plan A");
    assert_eq!(first_a.corrections.len(), 5);

    let cancelled = Arc::new(AtomicBool::new(true));
    assert!(filmcraft_media::cancel::with_cancel(&cancelled, || cached_plan(&analysis, &settings_b)).is_none());
    {
        let cache = plan_cache().lock();
        assert!(cache.entries.values().any(|entry| entry.plan.lock().is_none()), "cancelled B left a partial plan");
    }

    let retry_a = filmcraft_media::cancel::with_cancel(&not_cancelled, || cached_plan(&analysis, &settings_a)).expect("cached A");
    assert!(Arc::ptr_eq(&first_a, &retry_a));
    let state = *analysis.progress.lock();
    assert_eq!(state.stage, StabilizerStage::Complete);
    assert_eq!(state.completed_frames, state.total_frames);
}

#[test]
fn retained_budget_accounts_actual_plan_storage_and_evicts_variants() {
    let _guard = plan_test_guard();
    clear_stabilizer_cache();
    let analysis = plan_analysis(4);
    let settings = plan_settings(4.0);
    let mut plan = build_plan(&analysis.path, &settings);
    plan.meshes[0] = Some(MeshFrame::solve(128, 96, Affine::IDENTITY, &[], &[0.5; GRID_CELLS]).expect("actual mesh"));
    let plan = Arc::new(plan);
    let bytes = plan_retained_bytes(&plan);
    assert!(bytes > 1024, "actual plan storage was not measured");
    assert!(bytes < PLAN_CACHE_BUDGET);

    for variant in 0..=PLAN_CACHE_MAX_ENTRIES {
        let key = PlanKey { path: Arc::as_ptr(&analysis.path) as usize, smooth: variant as u64, crop: 0, safe: 0, maximum: 0, flags: 0 };
        let entry = cache_entry(&analysis, key);
        let retained = retain_plan(key, &entry, plan.clone());
        assert!(Arc::ptr_eq(&retained, &plan));
    }

    let cache = plan_cache().lock();
    assert!(cache.entries.len() <= PLAN_CACHE_MAX_ENTRIES);
    assert!(cache.retained_bytes <= PLAN_CACHE_BUDGET);
    let accounted = cache.entries.values().map(|entry| entry.retained_bytes.load(Ordering::Acquire)).sum::<usize>();
    assert_eq!(cache.retained_bytes, accounted);
}

#[test]
fn oversized_selected_plan_is_reused_instead_of_rebuilt_each_frame() {
    let _guard = plan_test_guard();
    clear_stabilizer_cache();
    let analysis = plan_analysis(4);
    let settings = plan_settings(4.0);
    let original = cached_plan(&analysis, &settings).expect("initial plan");
    let key = *plan_cache().lock().entries.keys().next().expect("selected key");
    clear_stabilizer_cache();
    let mut plan = build_plan(&analysis.path, &settings);
    // Reserved storage is part of the retained-byte accounting even before every frame is filled.
    plan.zooms.reserve(PLAN_CACHE_BUDGET / std::mem::size_of::<f64>() + 1);
    let plan = Arc::new(plan);
    assert!(plan_retained_bytes(&plan) > PLAN_CACHE_BUDGET);
    let entry = cache_entry(&analysis, key);
    retain_plan(key, &entry, plan.clone());
    drop(original);
    let reused = cached_plan(&analysis, &settings).expect("oversized selected plan");
    assert!(Arc::ptr_eq(&reused, &plan), "selected plan was evicted and rebuilt");
    assert_eq!(plan_cache().lock().entries.len(), 1);
    clear_stabilizer_cache();
}
