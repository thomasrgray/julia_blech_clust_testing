"""Regression checks for the confirmed animal/legacy stimulus conventions."""
import pytest
from lfp_session_labels import animal_for_session, canonicalize_stimulus_names


def test_parent_animal_and_consistent_embedded_id():
    assert animal_for_session('/media/thomas/Data/TG25/D1_retroortho2_260311_155032', {}) == 'TG25'
    assert animal_for_session('/data/TG22/Feb2_TG22_123', {}) == 'TG22'
    with pytest.raises(ValueError):
        animal_for_session('/data/TG25/Feb2_TG22_123', {})


def test_mapping_overrides_path_and_rejects_empty():
    path = '/data/TG25/session'
    assert animal_for_session(path, {'session': 'custom'}) == 'custom'
    assert animal_for_session(path, {path: 'fullpath'}) == 'fullpath'
    with pytest.raises(ValueError):
        animal_for_session(path, {'session': None})


def test_confirmed_legacy_names_map_simultaneously():
    raw = ['S', 'waterIA', 'waterBA', 'IA', 'BA', 'BK']
    assert canonicalize_stimulus_names(raw) == ['S', 'IA', 'BA', 'oIA', 'oBA', 'BK']
    assert raw[1] == 'waterIA'


def test_canonical_names_and_repeated_same_stimulus_remain():
    names = ['BA', 'IA', 'S', 'BK', 'oIA', 'oBA', 'IA']
    assert canonicalize_stimulus_names(names) == names
    assert canonicalize_stimulus_names(['waterIA', 'oIA']) == ['IA', 'oIA']


def test_ambiguous_alias_collision_cannot_pool_routes():
    with pytest.raises(ValueError, match='Ambiguous stimulus'):
        canonicalize_stimulus_names(['waterIA', 'IA', 'oIA'])


def test_zero_approved_channels_are_audited_before_stft(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    imports = ['DEFAULT_DATA_LIST', 'DEFAULT_GRAND_AVERAGE_DIR', 'PLOT_BASELINE_SECONDS',
               'PLOT_MAX_FREQ_HZ', 'SPECTROGRAM_END_MS', 'SPECTROGRAM_START_MS',
               'compute_stft', 'ephys_data', 'get_dataset_dirs', 'get_lfp_epoch_timing',
               'get_taste_names_in_lfp_order', 'load_cleaned_lfps',
               'normalize_trial_power', 'summarize_normalized_power']
    common = SimpleNamespace(**{key: None for key in imports})
    common.compute_stft = Mock()
    dat = SimpleNamespace(get_lfps=lambda: None, lfp_array_list=[object()])
    common.ephys_data = SimpleNamespace(ephys_data=lambda _: dat)
    common.get_taste_names_in_lfp_order = lambda _: ['waterIA', 'IA']
    common.get_lfp_epoch_timing = lambda _: {}
    common.load_cleaned_lfps = lambda *args: None
    monkeypatch.setitem(sys.modules, 'lfp_common', common)
    monkeypatch.setitem(sys.modules, 'validate_LFP_data', SimpleNamespace(
        validate_dataset=lambda *args, **kwargs: {'good_lfp_positions_by_region': {'GC': [], 'PC': []}}))
    spec = importlib.util.spec_from_file_location('grand_qc_test', Path(__file__).parents[1]/'grand_average_LFP.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit, data = {}, {}
    module.collect_session('/data/TG25/session', data, {'freq': None, 'relative_time': None}, audit)
    assert audit['status'] == 'excluded'
    assert audit['approved_channel_counts'] == {'GC': 0, 'PC': 0}
    assert audit['stimulus_names'] == {'waterIA': 'IA', 'IA': 'oIA'}
    assert data == {}
    common.compute_stft.assert_not_called()
