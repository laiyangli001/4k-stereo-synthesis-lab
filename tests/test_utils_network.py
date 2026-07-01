from utils import network


def test_configure_huggingface_endpoint_disables_symlink_usage(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS_WARNING", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    endpoint = network.configure_huggingface_endpoint(async_probe=False)

    assert endpoint == "https://hf-mirror.com"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert network.os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"
    assert network.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_configure_huggingface_endpoint_keeps_existing_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://custom.example")

    endpoint = network.configure_huggingface_endpoint(async_probe=False)

    assert endpoint == "https://custom.example"
    assert network.os.environ["HF_ENDPOINT"] == "https://custom.example"


def test_configure_huggingface_endpoint_default_does_not_block_on_probe(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    endpoint = network.configure_huggingface_endpoint()

    assert endpoint == "https://hf-mirror.com"
    assert network.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
