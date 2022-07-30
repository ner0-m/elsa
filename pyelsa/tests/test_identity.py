import pyelsa


def test_identity():
    print(dir(pyelsa))
    id = pyelsa.Identity((10, 10))


def test_identity2():
    id = pyelsa.identity.Identity((10, 10))


def test_identity3():
    from pyelsa import Identity

    id = Identity((10, 10))
    print(id)
