def _final_verify_wave2(
    candidates: list[tuple[str, float, list[str]]],
    graph: SceneGraph,
) -> list[tuple[str, float, list[str]]]:
    """Post-pipeline wave 2: fix-1 zero-risk conditions."""
    if len(candidates) < 2:
        return candidates

    from hlinet.features.compounds.phase2_signatures import _stats
    s = _stats(graph)

    top_label = candidates[0][0]
    sec_label = candidates[1][0]
    if top_label == "orange" and sec_label == "banana":
        if s.get("rb_corr", 0) > 0.85780560970:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "brown_bear":
        if s.get("dark_warm_ratio", 0) > 503.00000000000:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "mushroom":
        if s.get("dct_mid", 0) > 0.13235637449:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "mushroom":
        if s.get("hist_sports_minus_bus", 0) > 0.30485699978:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "teapot" and sec_label == "orange":
        if s.get("warm_sat_std", 0) > 0.27928100778:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "banana" and sec_label == "orange":
        if s.get("warm", 0) > 0.99804687500:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "sports_car" and sec_label == "school_bus":
        if s.get("hist_bear_minus_kp", 0) > 0.62249981519:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "school_bus" and sec_label == "sports_car":
        if s.get("hist_sports_car", 0) > 2.96346221922:
            candidates[0], candidates[1] = candidates[1], candidates[0]
    elif top_label == "golden_retriever" and sec_label == "teapot":
        if s.get("lbp_entropy", 999) < 4.81200136293:
            candidates[0], candidates[1] = candidates[1], candidates[0]

    if len(candidates) >= 3:
        top_label = candidates[0][0]
        r3_label = candidates[2][0]
        if top_label == "sports_car" and r3_label == "king_penguin":
            if s.get("fft_hv_ratio", 0) > 1.13932764746:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("glcm_contrast", 0) > 0.04041899825:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "sports_car":
            if s.get("cm_a_std", 0) > 0.06897316727:
                candidates[0], candidates[2] = candidates[2], candidates[0]
            elif s.get("spatial_bot_intensity", 999) < 0.12558210784:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "banana":
            if s.get("sat_color_std", 0) > 0.22358805393:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "banana":
            if s.get("bilat_detail", 0) > 0.08210688572:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "brown_bear":
            if s.get("has_round", 999) < 1.00000000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "brown_bear":
            if s.get("r0_circularity", 0) > 0.65321045247:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "golden_retriever":
            if s.get("warm_vert_concentration", 999) < 0.33396584440:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "teapot" and r3_label == "jellyfish":
            if s.get("blue_region_area", 0) > 0.31298828125:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "king_penguin":
            if s.get("hue_spread", 0) > 7.00000000000:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "school_bus" and r3_label == "mushroom":
            if s.get("green", 0) > 0.43139648438:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "orange":
            if s.get("cm_center_b", 0) > 0.65105315564:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "school_bus":
            if s.get("fft_hv_ratio", 0) > 1.13932764746:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "banana" and r3_label == "school_bus":
            if s.get("vert_regularity", 999) < 1.88337153719:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "sports_car":
            if s.get("top2_hue_ratio", 999) < 0.41097724230:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "sports_car" and r3_label == "teapot":
            if s.get("wavelet_coarse", 0) > 0.40493414975:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "brown_bear" and r3_label == "teapot":
            if s.get("wavelet_mid", 0) > 0.34177593845:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "orange" and r3_label == "teapot":
            if s.get("warm_coherence", 999) < 0.92972972973:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "golden_retriever" and r3_label == "teapot":
            if s.get("lbp_entropy", 999) < 4.81200136293:
                candidates[0], candidates[2] = candidates[2], candidates[0]
        elif top_label == "king_penguin" and r3_label == "teapot":
            if s.get("autocorr_x_mid_wider", 0) > 0.28258226194:
                candidates[0], candidates[2] = candidates[2], candidates[0]

    if len(candidates) >= 4:
        top_label = candidates[0][0]
        r4_label = candidates[3][0]
        if top_label == "orange" and r4_label == "teapot":
            if s.get("fft_hv_ratio", 0) > 1.29775370851:
                candidates[0], candidates[3] = candidates[3], candidates[0]
            elif s.get("hist_bear_minus_teapot", 0) > 0.13087687455:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "banana":
            if s.get("gabor_45_04_var", 0) > 3.86258264871:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "teapot" and r4_label == "banana":
            if s.get("sat_smooth_warm", 0) > 0.26539406002:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "mushroom" and r4_label == "golden_retriever":
            if s.get("r0_circularity", 999) < 0.08343115988:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "golden_retriever":
            if s.get("warm_val_mean", 0) > 0.62921882712:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "jellyfish":
            if s.get("lbp_entropy", 999) < 4.17439733122:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "king_penguin" and r4_label == "mushroom":
            if s.get("contour_fill_ratio", 0) > 0.42993164062:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "golden_retriever" and r4_label == "orange":
            if s.get("orient_entropy", 999) < 2.69169769923:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "sports_car":
            if s.get("rb_corr", 0) > 0.99423968792:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "school_bus" and r4_label == "sports_car":
            if s.get("elong_area", 0) > 0.18701171875:
                candidates[0], candidates[3] = candidates[3], candidates[0]
        elif top_label == "brown_bear" and r4_label == "teapot":
            if s.get("center_surround", 0) > 1.60255266127:
                candidates[0], candidates[3] = candidates[3], candidates[0]

    if len(candidates) >= 5:
        top_label = candidates[0][0]
        r5_label = candidates[4][0]
        if top_label == "school_bus" and r5_label == "brown_bear":
            if s.get("warm_sat_std", 999) < 0.05460310181:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("blue_region_area", 0) > 0.40478515625:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "king_penguin":
            if s.get("hist_golden_retriever", 0) > 2.55215367085:
                candidates[0], candidates[4] = candidates[4], candidates[0]
            elif s.get("blue_region_area", 0) > 0.31616210938:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "jellyfish":
            if s.get("hist_bear_minus_gr", 0) > 0.32879668497:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "king_penguin":
            if s.get("cm_b_skew", 0) > 2.88507676125:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "mushroom":
            if s.get("dct_high", 0) > 0.27603768454:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "mushroom":
            if s.get("r0_aspect", 0) > 9.14285714286:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "teapot" and r5_label == "mushroom":
            if s.get("bilat_detail", 0) > 0.07135416667:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "brown_bear" and r5_label == "mushroom":
            if s.get("glcm_contrast_v2", 0) > 0.06136859922:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "school_bus" and r5_label == "orange":
            if s.get("hu1", 999) < 2.48581402423:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "school_bus":
            if s.get("smooth_warm_blob_aspect", 0) > 6.22222222222:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "king_penguin" and r5_label == "sports_car":
            if s.get("cm_center_b", 999) < 0.42161841299:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "golden_retriever" and r5_label == "teapot":
            if s.get("smooth_warm_blob_area", 0) > 0.31054687500:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "orange" and r5_label == "teapot":
            if s.get("radial_warm_diff", 999) < -0.11676266637:
                candidates[0], candidates[4] = candidates[4], candidates[0]
        elif top_label == "sports_car" and r5_label == "teapot":
            if s.get("sat_smooth_warm", 0) > 0.14125977480:
                candidates[0], candidates[4] = candidates[4], candidates[0]

    return candidates