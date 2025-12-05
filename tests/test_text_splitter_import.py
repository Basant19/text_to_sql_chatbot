def test_recursive_character_text_splitter_import():
    """
    Quick CI check: ensure the required text-splitter package is importable.
    If this fails, your environment is missing langchain-text-splitters.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
    except Exception as e:
        # Raise an informative assertion error so CI logs are clear.
        raise AssertionError(
            "RecursiveCharacterTextSplitter could not be imported. "
            "Install the package with `pip install langchain-text-splitters` "
            "and ensure you're running tests in the correct virtualenv. "
            f"Original error: {e}"
        ) from e
