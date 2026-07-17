from prefscope.interpret.llm import LLMClient


class _Msg:
    def __init__(self, c): self.content = c
class _Choice:
    def __init__(self, c): self.message = _Msg(c)
class _Resp:
    def __init__(self, c): self.choices = [_Choice(c)]
class _Completions:
    def __init__(self, c): self._c = c; self.calls = []
    def create(self, **kw): self.calls.append(kw); return _Resp(self._c)
class _Chat:
    def __init__(self, c): self.completions = _Completions(c)
class _OpenAI:
    def __init__(self, c): self.chat = _Chat(c)


class _FlakyCompletions:
    """Raises `fail_n` times, then returns content."""
    def __init__(self, c, fail_n): self._c = c; self.fail_n = fail_n; self.n = 0
    def create(self, **kw):
        self.n += 1
        if self.n <= self.fail_n:
            raise RuntimeError("transient")
        return _Resp(self._c)
class _FlakyChat:
    def __init__(self, c, fail_n): self.completions = _FlakyCompletions(c, fail_n)
class _FlakyOpenAI:
    def __init__(self, c, fail_n): self.chat = _FlakyChat(c, fail_n)


def test_openai_retries_then_succeeds(monkeypatch):
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)  # no real delay
    fake = _FlakyOpenAI("recovered", fail_n=2)
    client = LLMClient(backend="openai", model="m", retries=3, _client=fake)
    assert client.raw([{"role": "user", "content": "hi"}]) == "recovered"
    assert fake.chat.completions.n == 3


def test_openai_raises_after_exhausting_retries(monkeypatch):
    import pytest
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    fake = _FlakyOpenAI("never", fail_n=99)
    client = LLMClient(backend="openai", model="m", retries=2, _client=fake)
    with pytest.raises(RuntimeError):
        client.raw([{"role": "user", "content": "hi"}])


class _Choice2:
    def __init__(self, c, fr): self.message = _Msg(c); self.finish_reason = fr
class _Resp2:
    def __init__(self, c, fr): self.choices = [_Choice2(c, fr)]
class _EmptyCompletions:
    def __init__(self, fr=None): self.n = 0; self.fr = fr; self.calls = []
    def create(self, **kw): self.n += 1; self.calls.append(kw); return _Resp2("", self.fr)
class _EmptyChat:
    def __init__(self, fr=None): self.completions = _EmptyCompletions(fr)
class _EmptyOpenAI:
    def __init__(self, fr=None): self.chat = _EmptyChat(fr)


def test_openai_reports_truncation_distinctly(monkeypatch):
    """An empty response with finish_reason=length is TRUNCATION, surfaced as such —
    not silently the same as a genuine empty (so it shows up in debug/logs)."""
    import pytest
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    fake = _EmptyOpenAI(fr="length")
    client = LLMClient(backend="openai", model="m", retries=2, _client=fake)
    with pytest.raises(Exception, match="truncated"):
        client.raw([{"role": "user", "content": "hi"}])


def test_reasoning_effort_is_forwarded(monkeypatch):
    fake = _EmptyOpenAI(fr="stop")
    client = LLMClient(backend="openai", model="m", retries=1,
                       reasoning_effort="minimal", _client=fake)
    try:
        client.raw([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    sent = fake.chat.completions.calls[0]
    assert sent.get("extra_body") == {"reasoning": {"effort": "minimal"}}


def test_openai_caps_total_requests_at_retries(monkeypatch):
    """Persistent empty responses must cost at most `retries` requests total —
    NOT retries * number_of_formats (the old 5*3=15 storm)."""
    import pytest
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    fake = _EmptyOpenAI()
    client = LLMClient(backend="openai", model="m", retries=3, _client=fake)
    with pytest.raises(Exception):
        # a schema request offers 3 formats; the cap must still be 3 requests, not 9
        client.raw([{"role": "user", "content": "hi"}],
                   response_schema={"type": "object"})
    assert fake.chat.completions.n == 3


class _CreditError(Exception):
    status_code = 402


class _DeadCompletions:
    def __init__(self): self.n = 0
    def create(self, **kw): self.n += 1; raise _CreditError("Insufficient credit")
class _DeadChat:
    def __init__(self): self.completions = _DeadCompletions()
class _DeadOpenAI:
    def __init__(self): self.chat = _DeadChat()


def test_openai_fast_fails_on_dead_key_no_retry_storm(monkeypatch):
    """A dead key / exhausted credit (402) must abort on the FIRST request, not retry —
    this is what turned ~1700 features into a ~26k-request storm."""
    import pytest
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    fake = _DeadOpenAI()
    client = LLMClient(backend="openai", model="m", retries=5, _client=fake)
    with pytest.raises(_CreditError):
        client.raw([{"role": "user", "content": "hi"}])
    assert fake.chat.completions.n == 1        # no retry on a non-recoverable error


def test_openai_backend_returns_text_no_json_mode_by_default():
    fake = _OpenAI('- "uses code blocks"')
    client = LLMClient(backend="openai", model="m", _client=fake)
    out = client.raw([{"role": "user", "content": "hi"}])
    assert out == '- "uses code blocks"'
    assert "response_format" not in fake.chat.completions.calls[0]
    assert fake.chat.completions.calls[0]["model"] == "m"


def test_claude_cli_backend_shells_out(monkeypatch):
    import subprocess

    class _Proc:
        returncode = 0
        stdout = "A"
        stderr = ""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    client = LLMClient(backend="claude-cli", model="claude-x")
    out = client.raw([{"role": "user", "content": "u"}])
    assert out == "A"


def test_claude_cli_retries_on_empty_output(monkeypatch):
    import subprocess
    import prefscope.interpret.llm as llm_mod
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    class _Proc:
        def __init__(self, rc, out): self.returncode = rc; self.stdout = out; self.stderr = ""

    def fake_run(*a, **k):
        calls["n"] += 1
        return _Proc(0, "") if calls["n"] == 1 else _Proc(0, "real concept")

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = LLMClient(backend="claude-cli", model="sonnet", retries=3)
    assert client.raw([{"role": "user", "content": "u"}]) == "real concept"
    assert calls["n"] == 2
