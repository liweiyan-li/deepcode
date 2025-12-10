"""
DeepCode Layout Manager
Organizes the visual structure using the Cyber components.
"""

from typing import Optional

import streamlit as st
from .components import (
    display_header,
    footer_component,
    input_method_selector,
    results_display_component,
    sidebar_control_panel,
    system_status_component,
)
from .styles import get_main_styles
from .handlers import (
    initialize_session_state,
    handle_start_processing_button,
    handle_error_display,
)


def setup_page_config():
    st.set_page_config(
        page_title="DeepCode",
        page_icon="assets/logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/deepcode",
            "About": "DeepCode AI Research Engine v3.0",
        },
    )


def main_layout():
    """Main layout execution"""
    # Initialize Core
    initialize_session_state()
    setup_page_config()

    # Inject Cyber Styles
    st.markdown(get_main_styles(), unsafe_allow_html=True)

    # Render Sidebar
    sidebar_control_panel()

    # Main Content Area
    display_header()

    # Determine Content State
    show_results = st.session_state.get("show_results", False)
    last_result = st.session_state.get("last_result", None)

    if show_results and last_result:
        results_display_component(last_result, st.session_state.task_counter)
        from .components import iteration_prompt_component, iteration_status_component
        iteration_prompt_component()
        iteration_status_component()
    else:
        # Landing State
        system_status_component()

        st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

        # Input Interface
        render_input_area()

    # Global Error Handler (Always active)
    handle_error_display()

    # Footer
    footer_component()

    return {}


def render_input_area():
    """Handles the logic for which input to show"""


    processing = st.session_state.get("processing", False)

    input_source: Optional[str] = None
    input_type: Optional[str] = None

    with st.container():
        input_source, input_type = input_method_selector(
            st.session_state.task_counter
        )

        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

        if input_source and not processing:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "START CODING 🚀", type="primary", use_container_width=True
                ):
                    handle_start_processing_button(input_source, input_type or "file")

        elif processing:
            st.markdown(
                """
                <div style="padding:1.5rem; border:1px solid var(--primary); border-radius:4px; background:rgba(0, 242, 255, 0.05); text-align:center;">
                    <div class="status-dot" style="display:inline-block; margin-right:10px;"></div>
                    <span style="font-family: var(--font-code); color: var(--primary); animation: pulse-glow 2s infinite;">NEURAL PROCESSING ACTIVE...</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif not input_source:
            st.markdown(
                """
                <div style="text-align:center; color:rgba(255,255,255,0.3); font-family:var(--font-code); font-size:0.8rem;">
                    AWAITING INPUT SIGNAL...
                </div>
                """,
                unsafe_allow_html=True,
            )
