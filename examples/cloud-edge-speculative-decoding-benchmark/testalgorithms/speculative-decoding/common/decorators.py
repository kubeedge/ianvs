"""Decorators that wrap user algorithm hooks with shared runtime behavior."""

from __future__ import annotations

from functools import wraps


def specdec_draft(user_fn):
    """Wrap a user-defined drafter `step` with common runtime behavior."""

    @wraps(user_fn)
    def wrapper(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        return self._run_draft_pipeline(
            session=session,
            feedback=feedback,
            max_draft_tokens=max_draft_tokens,
            user_step_fn=user_fn,
            extra_kwargs=kwargs,
        )

    return wrapper


def specdec_verify(user_fn):
    """Wrap a user-defined verifier `verify` with common runtime behavior."""

    @wraps(user_fn)
    def wrapper(self, session, draft_output=None, draft_ids=None, **kwargs):
        return self._run_verify_pipeline(
            session=session,
            draft_output=draft_output,
            draft_ids=draft_ids,
            user_verify_fn=user_fn,
            extra_kwargs=kwargs,
        )

    return wrapper
