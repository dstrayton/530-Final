            st.markdown('<div class="ff-alternative"><div class="ff-muted">Also consider</div>'
                        + material_picture(alternative) + f'<strong>{escape(alternative)}</strong></div>', unsafe_allow_html=True)
            def open_comparison():
                st.session_state["page"] = "Compare materials"
                st.session_state["compare_names"] = [primary, alternative]
                st.session_state.pop("comparison_selection", None)
            st.button("Compare options", on_click=open_comparison, use_container_width=True)
        st.write(answer["explanation"])
        st.caption("Spool artwork is illustrative. Colors do not indicate a brand or available product color.")


def main():
    import streamlit as st
    st.set_page_config(page_title="Filament Finder", page_icon=str(ASSETS / "logo.png") if (ASSETS / "logo.png").exists() else "🧵", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.container(key="topnav"):
        brand, navigation = st.columns([1, 1.6])
        with brand:
            st.markdown('<div class="ff-brand">' + picture("logo.png", "Filament Finder logo") + '<span>Filament Finder</span></div>', unsafe_allow_html=True)
        with navigation:
            with st.container(key="nav"):
                page = st.radio("Navigation", ["Find a material", "Compare materials", "About the data"],
                                horizontal=True, label_visibility="collapsed", key="page")
    st.markdown('<div class="ff-hero"><h1>Find the <span>right filament</span> for your project</h1>'
                '<p>Describe your part. Compare the evidence. Choose your material.</p></div>', unsafe_allow_html=True)
    try:
        performance, descriptions, reviews = load_data()
    except (OSError, ValueError):
        st.info("Add the prepared CSVs to the data folder, or run 01_prepare_data.ipynb first.")
        return

    if page == "About the data":
        with st.container(border=True, key="panel_about"):
            st.markdown('<div class="ff-heading">Good choices start with clear evidence.</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            cols[0].metric("Closed-printer experiments", len(performance))
            cols[1].metric("Material descriptions", len(descriptions))
            cols[2].metric("Cleaned customer reviews", len(reviews))
            st.write("The notebook cleans the three sources, removes open-printer experiments, and compares random forest regression with a material-average baseline. The selected method adds a predicted peak-stress column to the prepared data.")
            st.write("Recommendations combine fitted specimen-strength estimates, supplied descriptions, and product-specific customer reports. Materials with both experiments and descriptions are supported.")
            st.info("Data scope: closed-printer experiments only. No printer-type selection is needed; this does not establish performance on other printer types.")
            st.write("The app retrieves the closest observed settings for each material. Its final fitted values are estimates, not new measurements or independent test results. Use the notebook’s held-out results to evaluate model accuracy.")
            st.caption("The material-name alias PLA-GLOW assumes the supplied phosphor/phosphorus labels describe the same glow-in-the-dark material. Confirm with the original experimental source.")
            st.caption("Original source URLs and collection dates still need adding to the project documentation. Review examples are not a complete theme analysis. Open-printer rows are excluded from experimental modeling, not from the independently collected review text.")
    elif page == "Compare materials":
        with st.container(border=True, key="panel_compare"):
            st.markdown('<div class="ff-heading">Compare materials</div>', unsafe_allow_html=True)
            result = st.session_state.get("result")
            settings = st.session_state.get("submitted_settings", {"layer": 0.2, "infill": 50, "speed": 50, "pattern": "grid"})
            cards = build_evidence(settings)
            choices = list(cards)
            defaults = st.session_state.get("compare_names", ["PLA", "PETG"])
            defaults = [name for name in defaults if name in choices]
            names = st.multiselect("Choose two or more materials", choices, default=defaults, key="comparison_selection")
            st.caption(f'Requested settings: {settings["layer"]} mm layer · {settings["infill"]}% infill · {settings["speed"]} mm/s · {settings["pattern"]}. Change these in Find a material and submit a project to update them.')
            if len(names) < 2:
                st.info("Select at least two materials to see a comparison.")
            else:
                render_evidence(st, [cards[name] for name in names])
    else:
        left, right = st.columns([1, 1.65], gap="large")
        with left:
            with st.container(border=True, key="project_form"):
                st.markdown('<div class="ff-heading">Your project</div>', unsafe_allow_html=True)
                with st.form("project", border=False):
                    project = st.text_area("What are you making?", height=115,
                        placeholder="A rigid indoor electronics enclosure with a clean finish.")
                    st.markdown("**Your priorities**")
                    st.caption("1 = less important · 5 = most important")
                    priorities = {}
                    for label, default in [("Strength", 3), ("Stiffness", 3), ("Flexibility", 1), ("Surface finish", 4), ("Ease of printing", 4)]:
                        priorities[label] = st.slider(label, 1, 5, default, key="priority_" + label)
                    with st.expander("Print settings (optional)"):
                        layer = st.selectbox("Layer thickness (mm)", [0.1, 0.2, 0.3], index=1)
                        infill = st.selectbox("Infill (%)", [20, 50, 80], index=1)
                        speed = st.selectbox("Speed (mm/s)", [30, 50, 70], index=1)
                        pattern = st.selectbox("Infill pattern", ["grid", "triangles", "zigzag"])
                    submitted = st.form_submit_button("Recommend filament", use_container_width=True)
            if submitted:
                st.session_state.pop("result", None)
                if not project.strip():
                    st.warning("Describe your project first, then choose Recommend filament.")
                else:
                    settings = {"layer": layer, "infill": infill, "speed": speed, "pattern": pattern}
                    try:
                        with st.spinner("Comparing estimates and checking material evidence…"):
                            result = recommend(project + "\nPriority importance (1 low, 5 high): " + json.dumps(priorities), settings)
                            st.session_state["result"] = result
                            st.session_state["submitted_settings"] = settings
                            # Clear comparison widget state so Compare options uses this result.
                            st.session_state.pop("comparison_selection", None)
                    except Exception as error:
                        st.error(f"Could not produce a checked recommendation ({type(error).__name__}). Check your API settings and try again.")
        with right:
            result = st.session_state.get("result")
            render_recommendation(st, result)
            with st.container(border=True, key="panel_evidence"):
                st.markdown('<div class="ff-heading">Why this recommendation?</div>', unsafe_allow_html=True)
                if result and result["answer"]["status"] == "recommend":
                    answer = result["answer"]
                    cards = [result["evidence"][answer[key]] for key in ["recommended_material", "alternative_material"]]
                    render_evidence(st, cards)
                    if answer["limitations"]:
                        with st.expander("Tradeoffs and limitations", expanded=True):
                            for limitation in answer["limitations"]:
                                st.write("• " + limitation)
                    with st.expander("Source IDs and workflow details"):
                        st.json(result)
                    st.caption("Results reflect your last submitted project. Edit the inputs and submit again to update them.")
                else:
                    tabs = st.tabs(["Test data", "Material descriptions", "Customer reviews"])
                    for tab, text in zip(tabs, ["Compare predicted specimen peak stress and view the underlying measurements.",
                                               "Read the supplied material characteristics and application notes.",
                                               "Inspect product-specific printing experiences and their source IDs."]):
                        with tab:
                            st.info(text)
                    st.caption("Submit a project to see supporting evidence. No example result is presented as a real recommendation.")
    st.markdown('<div class="ff-footer">Filament Finder · A material selection study · Based on closed-printer experiments<br>'
                'Predictions, measured data, and customer reports are identified separately.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
