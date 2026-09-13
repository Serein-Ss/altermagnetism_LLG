from scripts.core.asset_structure import destination, LIVE


def test_uniform_literature():
    for paper in ('bauer_2011', 'hirst_mn2au_2022', 'laliena_crnb3s6_2020'):
        assert destination(f'assets/literature_reproduction/{paper}/r2_r5_analysis_20260910/figures/overview.png') == f'assets/literature_reproduction/{paper}/experiments/reproduction_validation/20260910_v1/figures/overview.png'


def test_live_paths_unchanged():
    for prefix in LIVE:
        assert destination(prefix+'manifest.json') == prefix+'manifest.json'


def test_research_and_idempotence():
    old = 'assets/research/runs/bauer_first_loop/20260913_v1/figures/physical_observables.png'
    new = 'assets/research/bauer_L25_first_loop/20260913_v1/model_comparison/physical_observables.png'
    assert destination(old) == new
    assert destination(new) == new


def test_bookkeeping_outside_assets():
    assert destination('assets/research/runs/zero_field_program/20260911_040025/tasks/000.json').startswith('logs/')
    assert destination('assets/research/runs/bauer_first_loop/20260913_v1/numba_cache/x.nbc').startswith('logs/cache/')
