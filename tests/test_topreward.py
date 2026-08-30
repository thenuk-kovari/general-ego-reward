from ego_progress.topreward import minmax


def test_minmax():
    assert minmax([-4.0, -3.0, -2.0]) == [0.0, 0.5, 1.0]


def test_minmax_constant():
    assert minmax([-2.0, -2.0]) == [0.5, 0.5]
