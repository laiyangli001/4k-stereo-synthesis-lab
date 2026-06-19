from pathlib import Path


def test_gui_stereo_preset_uses_dropdown_value_for_load_and_save():
    source = Path(__file__).resolve().parents[1] / "src" / "gui.py"
    text = source.read_text(encoding="utf-8")

    assert 'self.stereo_preset_dd.value = self._preset_to_display(cfg.get("Stereo Preset", DEFAULTS["Stereo Preset"]))' in text
    assert '"Stereo Preset": self._display_to_preset(self.stereo_preset_dd.value),' in text
    assert '"Stereo Preset": DEFAULTS["Stereo Preset"],' not in text
