from clean import ext_spt

def close_num():               #test simple float
    payload = {"c": 150.50}
    assert ext_spt(payload) == 150.50

def test_list():
    payload = {"c": [180.0, 181.0]}       # Test the "weird" list format
    assert ext_spt(payload) == 180.0

def test_missing():
    payload = {"other": 100}                   # Test what happens when the key is gone
    assert ext_spt(payload) is None

def test_empty_list():
    payload = {"c": []}                           # Test an empty list (edge case)
    assert ext_spt(payload) is None