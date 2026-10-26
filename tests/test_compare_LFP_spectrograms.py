import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from compare_LFP_spectrograms import (animal_for_session, cluster_test, clusters,
                                     figure_specs, make_figure, paired_animals, t_map)


def test_ten_figures_eighteen_tests():
    specs = figure_specs()
    assert len(specs) == 10
    assert sum(len(list(itertools.combinations(c, 2))) for _, c in specs) == 18
    assert dict(specs)['GC_retro'] == [('GC', 'BA'), ('GC', 'IA'), ('GC', 'S')]


def test_animal_ids_require_unambiguous_identity():
    assert animal_for_session('Feb2_TG22_123', {}) == 'TG22'
    assert animal_for_session('custom', {'custom': 'rat1'}) == 'rat1'
    with pytest.raises(ValueError):
        animal_for_session('custom', {})


def test_complete_session_pairing_and_equal_animal_weight():
    data = {('gc', 'BA'): {'sessions': ['a_TG1_x', 'b_TG1_x', 'a_TG2_x', 'missing_TG3_x'],
                            'spectrograms': [np.full((2, 3), x) for x in (1, 3, 8, 99)]},
            ('PC', 'BA'): {'sessions': ['a_TG2_x', 'b_TG1_x', 'a_TG1_x'],
                            'spectrograms': [np.full((2, 3), x) for x in (7, 2, 0)]}}
    arrays, animals, grouped, excluded = paired_animals(data, [('GC','BA'), ('PC','BA')], {})
    assert animals == ['TG1', 'TG2']
    np.testing.assert_allclose(arrays[0][:, 0, 0], [2, 8])
    np.testing.assert_allclose(arrays[1][:, 0, 0], [1, 7])
    assert excluded == ['missing_TG3_x']


def test_exact_null_matches_all_sign_flips():
    d = np.random.default_rng(4).normal(size=(5, 3, 4)) + 1
    result = cluster_test(d)
    full = []
    for signs in itertools.product((-1, 1), repeat=5):
        full.append(max((mass for _, mass, _ in clusters(
            t_map(d*np.array(signs)[:,None,None]), result['threshold'])), default=0))
    np.testing.assert_allclose(np.sort(full), np.repeat(np.sort(result['null_max_mass']), 2))
    for c in result['clusters']:
        assert c['p_cluster'] == pytest.approx(np.mean(np.array(full) >= c['mass']*(1-1e-12)))


def test_zero_and_small_sample_cannot_claim_significance():
    result = cluster_test(np.zeros((6, 3, 5)))
    assert not result['significant'].any()
    result = cluster_test(np.ones((6, 3, 5)))
    assert not result['significant'].any()
    assert result['minimum_family_p'] == 18/32


def test_detect_large_consistent_effect_and_sign_reversal():
    d = np.random.default_rng(2).normal(3, .1, (10, 3, 5))
    positive = cluster_test(d)
    negative = cluster_test(-d)
    assert positive['significant'].all()
    np.testing.assert_array_equal(positive['p_family'], negative['p_family'])
    assert positive['clusters'][0]['sign'] == 1
    assert negative['clusters'][0]['sign'] == -1


def test_monte_carlo_reproducibility_and_invalid_inputs():
    d = np.random.default_rng(3).normal(size=(12, 3, 5))
    a, b = cluster_test(d, 30, 2), cluster_test(d, 30, 2)
    assert not a['exact']
    np.testing.assert_array_equal(a['null_max_mass'], b['null_max_mass'])
    for bad in (np.ones((1, 3, 5)), np.full((4, 3, 5), np.nan)):
        with pytest.raises(ValueError):
            cluster_test(bad)


def test_render_two_and_three_conditions(tmp_path):
    rng = np.random.default_rng(3)
    arrays = [rng.normal(size=(4, 4, 12)) for _ in range(3)]
    args = SimpleNamespace(permutations=20, seed=0, alpha=.05)
    for count in (2, 3):
        results = make_figure(f'test{count}', [('GC', str(i)) for i in range(count)],
                             arrays[:count], ['a','b','c','d'], np.arange(1,5),
                             np.linspace(-1,1.75,12), np.arange(12)>=4, tmp_path, args)
        assert len(results) == (1 if count == 2 else 3)
        assert (tmp_path / f'test{count}.png').exists()


def test_main_integration_ten_figures(monkeypatch, tmp_path):
    """Exercise CLI, session collection, pairing, reports and all 18 outputs."""
    import json
    import sys
    from compare_LFP_spectrograms import main
    def collect(path, data, reference, audit=None):
        if path == "a_TG3_x":
            audit.update(status="excluded", reason="no approved LFP channels in any region")
            return
        audit["status"] = "included"
        reference['freq'] = np.arange(0., 5.)
        reference['relative_time'] = np.linspace(-1, 1.75, 12)
        rng = np.random.default_rng(int(path.split('_')[1][2:]))
        for _, conditions in figure_specs():
            for condition in conditions:
                data[condition] = {'spectrograms': [rng.normal(size=(5, 12))],
                                   'sessions': [path]}
    monkeypatch.setitem(sys.modules, 'grand_average_LFP', SimpleNamespace(collect_session=collect))
    monkeypatch.setitem(sys.modules, 'lfp_common', SimpleNamespace(
        get_dataset_dirs=lambda paths, _: paths, DEFAULT_DATA_LIST='unused',
        PLOT_MAX_FREQ_HZ=35, SPECTROGRAM_START_MS=-1000, SPECTROGRAM_END_MS=1750))
    out = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['compare_LFP_spectrograms.py', '--dataset',
                                    'a_TG1_x', '--dataset', 'a_TG2_x', '--dataset', 'a_TG3_x',
                                    '--output-dir', str(out)])
    main()
    assert len(list(out.glob('*.png'))) == 10
    assert len(list(out.glob('*.npz'))) == 18
    report = json.loads((out/'statistics.json').read_text())
    assert all(f['status']=='tested' for f in report['figures'].values())
    assert report['session_audit']['a_TG3_x']['status'] == 'excluded'
    assert report['session_audit']['a_TG1_x']['status'] == 'included'
