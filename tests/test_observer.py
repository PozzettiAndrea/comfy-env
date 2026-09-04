"""Contract: the optional observer is safe precisely because it holds nothing.

The model proxy this replaces claimed to hold memory, so ComfyUI relied on
its numbers and reached deeper into it; both of comfy-env's historical loud
breaks were new attribute reads landing on that object. This one answers
zero to everything, so no caller depends on it.
"""

from comfy_env.isolation import observer as O


class TestSwitch:
    def test_off_unless_explicitly_turned_on(self):
        """Catches: defaulting to on. The floor needs nothing from this, and
        it is the one remaining piece with a breakage history."""
        assert O.enabled({}) is False
        assert O.enabled({O.ENV_VAR: ""}) is False
        assert O.enabled({O.ENV_VAR: "0"}) is False
        assert O.enabled({O.ENV_VAR: "off"}) is False
        assert O.enabled(None) is False

    def test_the_usual_affirmatives_all_work(self):
        for value in ("1", "on", "true", "YES", " On "):
            assert O.enabled({O.ENV_VAR: value}) is True, value


class TestSentinel:
    def test_the_free_button_is_recognised(self):
        assert O.is_free_all(1e30) is True

    def test_it_is_a_threshold_not_an_equality(self):
        """Catches: comparing against 1e30 exactly. Upstream computes the
        value and float arithmetic has already altered it once in transit."""
        assert O.is_free_all(1e30 * 0.999) is True
        assert O.is_free_all(9.9e29) is True

    def test_ordinary_pressure_is_not_the_button(self):
        """Catches: a threshold low enough that a real eviction request looks
        like the user pressed Free, which would release every worker mid
        workflow."""
        for real in (0, 1024, 24 * 1024 ** 3, 512 * 1024 ** 3):
            assert O.is_free_all(real) is False, real

    def test_junk_is_not_the_button(self):
        assert O.is_free_all(None) is False
        assert O.is_free_all("lots") is False


class TestObserverHoldsNothing:
    def test_every_size_question_answers_zero(self):
        """The property that makes loose duck typing honest here: ComfyUI
        asks what we hold, gets nothing, and moves on to something it can
        actually evict."""
        obs = O.MemoryObserver()
        assert obs.model_memory() == 0
        assert obs.model_offloaded_memory() == 0
        assert obs.loaded_size() == 0
        assert obs.model_size() == 0
        assert obs.is_dynamic() is False
        assert obs.is_dead() is False

    def test_unloading_declines_rather_than_lying(self):
        """Catches: returning True, which would tell ComfyUI it had recovered
        memory that never existed and stop the eviction loop early."""
        assert O.MemoryObserver().model_unload(1024) is False

    def test_unknown_attributes_do_not_raise(self):
        """This is the whole failure mode of the old proxy: a new upstream
        attribute read landed on it and raised inside ComfyUI's eviction
        loop. Here it must be harmless."""
        obs = O.MemoryObserver()
        assert obs.some_future_upstream_method() is None
        assert obs.another(1, 2, keyword=3) is None

    def test_dunders_still_raise(self):
        """Catches: swallowing __deepcopy__ and friends, which makes copy,
        pickle and iteration behave bizarrely instead of failing loudly.

        Only names object does NOT define: __getstate__ is inherited from
        object on 3.11+, so it resolves on the type and never reaches
        __getattr__ at all.
        """
        obs = O.MemoryObserver()
        for name in ("__deepcopy__", "__iter__", "__len__"):
            try:
                getattr(obs, name)
            except AttributeError:
                continue
            raise AssertionError(name + " was swallowed")


class TestSignals:
    def test_the_button_fires_the_free_all_callback_only(self):
        fired = []
        obs = O.MemoryObserver(on_free_all=lambda: fired.append("free"),
                               on_pressure=lambda n: fired.append(("pressure", n)))
        obs.model_unload(1e30)
        assert fired == ["free"]

    def test_ordinary_pressure_fires_the_pressure_callback_only(self):
        fired = []
        obs = O.MemoryObserver(on_free_all=lambda: fired.append("free"),
                               on_pressure=lambda n: fired.append(("pressure", n)))
        obs.model_unload(8 * 1024 ** 3)
        assert fired == [("pressure", 8 * 1024 ** 3)]

    def test_a_raising_callback_cannot_escape_into_comfyui(self):
        """Catches: letting our own bug take down the host's eviction loop.
        Being a listener must never be able to break the thing listened to."""
        def boom(*args):
            raise RuntimeError("bug in our handler")
        obs = O.MemoryObserver(on_free_all=boom, on_pressure=boom)
        assert obs.model_unload(1e30) is False
        assert obs.model_unload(1024) is False
