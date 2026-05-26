"""
Login page for the backtester Streamlit UI.

Exposes a single public function :func:`login_page` that renders the
complete sign-in form. Intended to be called from the main app router
before any protected page is shown.

No global DB session is held — a fresh session is opened inline inside
the button handler using :func:`~ui.db.get_db`.
"""

from __future__ import annotations

import streamlit as st

import ui.auth as auth
from ui.db import get_db, get_user_by_email

# ---------------------------------------------------------------------------
# CSS — injected once at call time via st.markdown
# ---------------------------------------------------------------------------

_LOGIN_CSS = """
<style>
/* Login page — no hardcoded background so dark mode works correctly */
.login-spacer { height: 6vh; }
</style>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def login_page() -> None:
    """
    Render the centered sign-in card.

    Layout
    ------
    * Centred via ``st.columns([1, 2, 1])``.
    * Title: ``"⚡ Backtester"`` + subtitle caption.
    * Email and password inputs.
    * "Sign In" primary button.
    * Error banner on bad credentials.
    * Muted hint text at the bottom.

    Behaviour
    ---------
    On successful login:

    1. Opens a DB session with :func:`~ui.db.get_db`.
    2. Looks up the user by email.
    3. Verifies the password via :func:`~ui.auth.verify_password`.
    4. Calls :func:`~ui.auth.login` to write auth info to ``session_state``.
    5. Calls ``st.rerun()`` to reload the app in authenticated state.

    On failed login, shows ``st.error("Invalid email or password")``.
    """
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    # Vertical breathing room above the card
    st.markdown('<div class="login-spacer"></div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])

    with col:
        with st.container(border=True):
            st.markdown("## ⚡ Backtester")
            st.caption("Quantitative Research Platform")
            st.divider()

            email: str = st.text_input(
                "Email", placeholder="you@company.com", label_visibility="visible"
            )
            password: str = st.text_input(
                "Password", type="password", placeholder="••••••••",
                label_visibility="visible"
            )
            st.markdown("")  # small spacer

            if st.button("Sign In", use_container_width=True, type="primary"):
                _handle_sign_in(email, password)

            st.caption("Contact your admin to create an account")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _handle_sign_in(email: str, password: str) -> None:
    """
    Attempt to authenticate the user and update session state.

    Opens a database session, verifies credentials, and either calls
    ``st.rerun()`` on success or ``st.error(...)`` on failure. The DB
    session is fully closed before ``st.rerun()`` is called.

    Parameters
    ----------
    email:
        Email address entered by the user.
    password:
        Plain-text password entered by the user.
    """
    success = False

    with get_db() as db:
        user = get_user_by_email(db, email)
        if user is not None and auth.verify_password(password, user.password_hash):
            # Write session_state while the ORM session is still open so
            # that lazy-loaded attributes remain accessible.
            auth.login(user)
            success = True

    if success:
        st.rerun()
    else:
        st.error("Invalid email or password")
