    )
    mono_data = []
    for col in FEATURE_COLUMNS:
        info = FEATURE_DEFINITIONS[col]
        direction = "Decreases Risk (↓)" if info["monotone"] == -1 else "Increases Risk (↑)"
        mono_data.append({
            "Feature Name": col,
            "Direction": direction,
            "Description": info["description"],
        })
    st.dataframe(pd.DataFrame(mono_data), use_container_width=True)
    st.markdown("---")
    st.subheader("3. Regulatory Compliance Checklist")
    st.markdown(
        """
        - ✅ **UIDAI Aadhaar Data Vault Regulations:** 12-digit Aadhaar numbers quarantined in vault. Zero Aadhaar in ML dataset.
        - ✅ **DPDP Act 2023 (Data Minimization):** Street addresses stripped; only coarse postal risk tier retained.
        - ✅ **RBI Master Direction on KYC:** Deterministic validation rules enforce mandatory pass/fail gate before underwriting.
        - ✅ **RBI Digital Lending Guidelines (DLG):** Local TreeSHAP reason codes explain all adverse decisions without demographic proxies.
        """
    )
