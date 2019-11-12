""" Generate 5000 points in two concentric circles then analyze and plotthe topology using kmapper """
""" Ploting and stringifying helper functions (utilities) """


def argify(kwarg_dict=None, **kwargs):
    """ Return a string expression equivalent of the ** operator on a dict (k=v arguments)

    >>> argify(**{'hello': 2, 'bye': []})
    'hello=2, bye=[]'
    >>> argify({'hello': 42, 'bye': {}})
    'hello=42, bye={}'
    """
    kwarg_dict = kwarg_dict or {}
    kwarg_dict.update(kwargs)
    return ', '.join(f'{k}={v}' for k, v in kwarg_dict.items())
