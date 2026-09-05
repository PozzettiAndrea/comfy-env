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


class TestSurvivesTheLedger:
    """The entry has to survive ComfyUI's own housekeeping. Every attribute
    below is one the manager reads on entries, and the catch-all answers a
    CALLABLE, which is the wrong answer for all of them."""

    def test_it_is_not_pruned_when_a_host_model_is_collected(self):
        """Catches THE defect that made the observer useless: cleanup_models
        pops every entry whose real_model() is None, and __getattr__ handed
        back a function returning None. It fired the first time any host
        model was garbage collected, which is precisely when the Free button
        or an OOM had just run."""
        obs = O.MemoryObserver()
        assert obs.real_model() is not None

    def test_currently_used_cannot_be_flipped(self):
        """Catches: an attribute a later upstream write can set True.
        loaded_models(only_currently_used=True) hands entries straight back
        into load_models_gpu (controlnet.py, three extras nodes), which would
        run the whole admission path against this object."""
        obs = O.MemoryObserver()
        assert obs.currently_used is False
        obs.currently_used = True
        assert obs.currently_used is False

    def test_it_is_never_a_clone_of_anything(self):
        """Catches: clone_base_uuid left to __getattr__ (protection by
        accident) or set to None. unload_model_and_clones keeps entries whose
        uuid differs from the target's; None would MATCH a target whose own
        uuid is None and the observer would be asked to free everything for a
        single model eviction."""
        obs = O.MemoryObserver()
        assert obs.clone_base_uuid is not None
        assert obs.clone_base_uuid != "any-uuid"
        assert obs.is_clone(object()) is False

    def test_cloning_it_raises_instead_of_returning_none(self):
        """Catches: multigpu.py calling .clone() on list entries and using
        the None the catch-all would return."""
        obs = O.MemoryObserver()
        try:
            obs.clone()
        except RuntimeError:
            return
        raise AssertionError("clone() returned instead of raising")

    def test_the_finalizer_can_be_detached(self):
        """Catches: model_finalizer.detach() landing on the catch-all, which
        works by luck; upstream calls it when it drops an entry."""
        obs = O.MemoryObserver()
        obs.model_finalizer.detach()

    def test_devices_are_real_values_not_callables(self):
        """Catches: load_device answering a function, which multigpu.py
        compares against a torch device."""
        obs = O.MemoryObserver(device="cuda:0")
        assert obs.load_device == "cuda:0"
        assert obs.offload_device == "cuda:0"


class TestDisableSmartMemory:
    """Under --disable-smart-memory every request is 1e32, so the sentinel
    stops meaning "the button"."""

    def test_ordinary_pressure_is_not_read_as_the_button(self):
        """Catches: the shipped threshold, which under that flag reads EVERY
        host load as the Free button and broadcasts a full release each time."""
        assert O.is_free_all(8 * 1024 ** 3, smart_memory=True) is False

    def test_with_the_flag_every_request_means_release(self):
        """Catches: trying to tell them apart. The flag means "forget every
        model after each run", so release is the honest answer to all of
        them, and the pressure callback must not also fire."""
        assert O.is_free_all(8 * 1024 ** 3, smart_memory=False) is True
        assert O.is_free_all(1e32, smart_memory=False) is True

    def test_the_flag_is_read_from_comfyui_not_guessed(self, monkeypatch):
        """Catches: hardcoding smart memory on. The observer must ask the
        host, because the flag is the host's."""
        import sys
        import types
        mm = types.ModuleType("comfy.model_management")
        mm.DISABLE_SMART_MEMORY = True
        pkg = types.ModuleType("comfy")
        pkg.model_management = mm
        monkeypatch.setitem(sys.modules, "comfy", pkg)
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
        fired = []
        obs = O.MemoryObserver(on_free_all=lambda: fired.append("free"),
                               on_pressure=lambda n: fired.append("pressure"))
        obs.model_unload(8 * 1024 ** 3)
        assert fired == ["free"]
