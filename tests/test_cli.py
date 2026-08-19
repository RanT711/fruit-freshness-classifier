def test_package_can_be_imported():
    import fruit_grader

    assert fruit_grader.__name__ == "fruit_grader"


def test_prepare_parser_uses_default_seed():
    from scripts.prepare_dataset import build_parser

    args = build_parser().parse_args(["--raw-dir", "raw", "--output-dir", "processed"])

    assert args.seed == 42
    assert args.split == "70,15,15"
