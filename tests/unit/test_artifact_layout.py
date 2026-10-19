from scripts.core.artifact_layout import destination


def test_categories():
    assert destination('README.md') == 'README.md'
    assert destination('GUIDE/plan.md') == 'GUIDE/plan.md'
    assert destination('output/run/report.md') == 'docs/reports/output/run/report.md'
    assert destination('output/literature_reproduction/bauer_2011/r/a.h5') == 'data/literature_reproduction/bauer_2011/runs/r/a.h5'
    assert destination('output/bauer_first_loop/r/a.pt') == 'output/bauer_first_loop/r/a.pt'
    assert destination('data/literature_reproduction/bauer_2011/a.h5') == 'data/literature_reproduction/bauer_2011/a.h5'


def test_no_prefix_substring_replacement():
    assert destination('conf/literature_reproduction/physics/a.yaml') == 'conf/literature_reproduction/physics/a.yaml'
