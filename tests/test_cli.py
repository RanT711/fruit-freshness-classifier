def test_package_can_be_imported():
    import fruit_grader

    assert fruit_grader.__name__ == "fruit_grader"
