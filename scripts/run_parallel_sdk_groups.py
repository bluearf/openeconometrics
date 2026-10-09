"""Run two SDK pytest children inside one unchanged host wall deadline.

Raw child evidence keeps its execution order. The merged files contain the actual
testcase elements and phase records, ordered by the original selector contract.
Legacy mode covers the complete suite on one host. Distributed mode owns two of
four source-pinned node partitions and explicitly records a partial shard; only
the separate actual-job-bound aggregate can certify the whole 900-second cohort.
No failed, skipped or interrupted selected test is accepted.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXPENSIVE = ("tests/test_econ_saved_prediction_linear.py",
             "tests/test_control_function_common_prediction.py",
             "tests/test_streaming_control_function_engine.py",
             "tests/test_control_stream_acceptance.py")
# Advisory measured wall durations only influence ordering. All future node IDs
# receive the positive fallback and remain in the complete source collection.
# The literal profile is pinned with the tested source, never read from old CI
# artifacts at runtime; it is not a test selection or acceptance criterion.
DEFAULT_NODE_SECONDS = 0.05
ADVISORY_NODE_SECONDS = {
    "packages/openecon-charts/tests/test_chart_network.py::test_duplicate_ids_bounded_lists_and_aggregate_utf8_label_budget": 1.753814,
    "packages/openecon-charts/tests/test_chart_network_workbench.py::test_timeline_aggregate_caps_and_label_memory_budget_are_enforced": 1.85096,
    "packages/openecon-charts/tests/test_network_export.py::test_font_failure_does_not_replace_existing_output": 1.481194,
    "packages/openecon-charts/tests/test_network_export.py::test_real_offline_exports_and_exclusive_publication[pdf]": 3.017688,
    "packages/openecon-charts/tests/test_network_export.py::test_real_offline_exports_and_exclusive_publication[png]": 1.443853,
    "packages/openecon-charts/tests/test_network_export.py::test_real_offline_exports_and_exclusive_publication[svg]": 1.341706,
    "packages/openecon-charts/tests/test_network_export.py::test_reproducible_seed_layout_and_saved_camera": 4.437458,
    "packages/openecon-charts/tests/test_network_export.py::test_work_refusal_does_not_report_success": 1.43206,
    "tests/test_ca_uncertainty.py::test_count_sampling_exact_constraints_and_independent_probability_law[multinomial]": 3.879388,
    "tests/test_ca_uncertainty.py::test_count_sampling_exact_constraints_and_independent_probability_law[row_multinomial]": 3.871692,
    "tests/test_canon_frequency_uncertainty.py::test_complete_199_draw_literal_expansion_generalized_eigen_oracle[coefficients-normal]": 1.286746,
    "tests/test_canon_frequency_uncertainty.py::test_complete_199_draw_literal_expansion_generalized_eigen_oracle[coefficients-skew_t8]": 1.266152,
    "tests/test_canon_uncertainty.py::test_complete_json_replay_typed_identities_missing_alignment_and_seed[coefficients]": 1.028717,
    "tests/test_canon_uncertainty.py::test_two_domains_full_199_draw_generalized_eigen_oracle[coefficients-normal]": 1.34264,
    "tests/test_canon_uncertainty.py::test_two_domains_full_199_draw_generalized_eigen_oracle[coefficients-skew_t8]": 1.29952,
    "tests/test_categorical_frequency_pca.py::test_count_missing_zero_and_duplicate_index_preserve_physical_positions": 1.386917,
    "tests/test_categorical_frequency_regression.py::test_delivery_fixture_all_four_full_restore": 1.036257,
    # Complete historical public-API 8,100-law duration; advisory scheduling only.
    "tests/test_causal_multiarm_neyman.py::test_two_strata_full_cartesian_8100_law_full_covariance_and_population_weights": 809.884951728,
    "tests/test_causal_neyman.py::test_complete_universe_unbiasedness_full_covariance_bound_and_heterogeneity_gap[False-8-3]": 1.717712,
    "tests/test_causal_neyman.py::test_complete_universe_unbiasedness_full_covariance_bound_and_heterogeneity_gap[True-8-3]": 1.803336,
    "tests/test_causal_neyman.py::test_stratified_cartesian_universe_size_weights_and_full_bound": 2.454832,
    "tests/test_control_function_common_prediction.py::test_complete_disk_predictions_missing_typed_index_semantic_replay_once[gaussian_hc0]": 2.731463,
    "tests/test_control_function_stream_state.py::test_admission_precedes_source_and_ambient_state_is_preserved": 1.164114,
    "tests/test_control_function_stream_state.py::test_coherently_regenerated_nonstationary_outcome_refuses_without_refit": 3.339987,
    "tests/test_control_function_stream_state.py::test_compact_helper_retains_joint_jacobians_and_link_intervals": 1.750251,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfcloglog]": 5.154218,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cffraclogit]": 3.114186,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfgamma]": 2.932332,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfinvgauss]": 3.24092,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cflogit]": 5.176022,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfpoisson]": 3.082743,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfprobit]": 5.263119,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[False-cfregress]": 2.621306,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfcloglog]": 5.555669,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cffraclogit]": 3.303248,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfgamma]": 3.102062,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfinvgauss]": 3.568192,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cflogit]": 5.537608,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfpoisson]": 3.087828,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfprobit]": 5.548565,
    "tests/test_control_function_stream_state.py::test_complete_reload_and_common_targets_without_fit[True-cfregress]": 2.611724,
    "tests/test_control_function_stream_state.py::test_factory_reload_needs_explicit_source_and_compact_rows_are_absent": 5.41961,
    "tests/test_control_function_stream_state.py::test_physical_cache_cannot_admit_changed_execution_provenance": 8.560132,
    "tests/test_control_function_stream_state.py::test_physical_cache_reads_all_bytes_and_all_reporting_before_hit": 16.761173,
    "tests/test_control_function_stream_state.py::test_physical_cache_rehashed_state_and_forged_stat_identity_refuse": 23.471848,
    "tests/test_control_function_stream_state.py::test_physical_json_auto_reload_changed_source_and_explicit_relocation[csv]": 20.305905,
    "tests/test_control_function_stream_state.py::test_physical_json_auto_reload_changed_source_and_explicit_relocation[parquet]": 20.592203,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cfcloglog]": 3.252009,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cffraclogit]": 2.081386,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cfgamma]": 2.130706,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cfinvgauss]": 2.489044,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cflogit]": 3.34168,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cfpoisson]": 1.991893,
    "tests/test_control_function_stream_state.py::test_refinement_allows_one_ulp_parameters_and_rebatched_replay[cfprobit]": 3.392998,
    "tests/test_control_function_stream_state.py::test_rehashed_optimizer_history_and_refinement_refuse[concavity]": 1.064592,
    "tests/test_control_function_stream_state.py::test_rehashed_optimizer_history_and_refinement_refuse[iterations]": 1.033846,
    "tests/test_control_function_stream_state.py::test_rehashed_optimizer_history_and_refinement_refuse[scaled_gradient]": 1.061207,
    "tests/test_control_function_stream_state.py::test_rehashed_optimizer_history_and_refinement_refuse[step]": 1.038017,
    "tests/test_control_function_stream_state.py::test_single_list_cluster_saved_semantics_replay": 1.111385,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-cloglog]": 49.63497,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-fractional_logit]": 51.340859,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-gamma]": 33.103418,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-gaussian]": 30.122127,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-logit]": 48.454331,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-poisson]": 36.104008,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[cluster-probit]": 55.041561,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-cloglog]": 45.566799,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-fractional_logit]": 44.938951,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-gamma]": 30.39042,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-gaussian]": 28.29709,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-inverse_gaussian]": 39.597048,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-logit]": 45.057824,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-poisson]": 33.24346,
    "tests/test_control_stream_acceptance.py::test_public_streamed_fit_beyond_5000_matches_independent_full_sandwich[robust-probit]": 50.538287,
    "tests/test_dataset.py::test_csv_projection_unknown_count_and_replay": 1.106502,
    "tests/test_dataset.py::test_parquet_projection_keeps_stored_index_without_exposing_storage_columns[index0]": 1.809623,
    "tests/test_dataset.py::test_parquet_projection_keeps_stored_index_without_exposing_storage_columns[index1]": 1.733967,
    "tests/test_dataset.py::test_parquet_projection_keeps_stored_index_without_exposing_storage_columns[index2]": 1.788748,
    "tests/test_dataset.py::test_parquet_projection_nullable_types_and_replay": 1.146472,
    "tests/test_dataset.py::test_parquet_projection_preserves_declared_category_order_and_unused_levels": 1.177673,
    "tests/test_dataset.py::test_scan_reads_only_metadata_and_head_remains_bounded": 1.163778,
    "tests/test_dependent_meta_oracles.py::test_complete_whole_study_deletions_against_independent_refits[effect]": 1.989685,
    "tests/test_dependent_meta_oracles.py::test_complete_whole_study_deletions_against_independent_refits[study]": 2.106261,
    "tests/test_dependent_meta_oracles.py::test_plain_checksum_tamper_rejected_and_no_runtime_oracle_imports": 1.524382,
    "tests/test_desktop_runtime.py::test_cached_import_validates_checksum_and_reuses_snapshot": 4.00355,
    "tests/test_desktop_runtime.py::test_desktop_worker_does_not_inherit_cloud_or_auth_environment": 2.545503,
    "tests/test_desktop_runtime.py::test_entry_announces_ephemeral_port_and_stdin_shutdown[eof]": 3.936199,
    "tests/test_desktop_runtime.py::test_entry_announces_ephemeral_port_and_stdin_shutdown[shutdown]": 4.314239,
    "tests/test_desktop_runtime.py::test_explicit_close_cleans_running_worker_and_keeps_history": 2.594515,
    "tests/test_desktop_runtime.py::test_malformed_worker_messages_stop_session_without_unpickling[_invalid_result_worker]": 1.196905,
    "tests/test_desktop_runtime.py::test_malformed_worker_messages_stop_session_without_unpickling[_oversized_worker]": 1.196563,
    "tests/test_desktop_runtime.py::test_malformed_worker_messages_stop_session_without_unpickling[_pickle_worker]": 1.206683,
    "tests/test_desktop_runtime.py::test_persistent_torch_session_switch_and_project_tokens": 5.477646,
    "tests/test_desktop_runtime.py::test_result_outbox_total_size_is_bounded_and_failure_preserves_pending": 1.109176,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[bsqreg]": 2.647533,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[iqreg]": 2.866085,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[mixed]": 2.771654,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[prais]": 2.696207,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[qreg]": 2.646871,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[rreg]": 2.837165,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[sqreg]": 2.757868,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtfmb]": 2.615593,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtgls]": 2.713302,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtivreg_re]": 2.766821,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtpcse]": 2.799138,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtreg_pooled]": 2.912,
    "tests/test_econ_saved_prediction_linear.py::test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index[xtreg_re]": 2.817054,
    "tests/test_econ_streaming_prediction.py::test_global_effects_gradients_and_covariance_match_dense[ame-ologit]": 1.138941,
    "tests/test_econ_streaming_prediction.py::test_global_effects_gradients_and_covariance_match_dense[ame-ols]": 1.128311,
    "tests/test_econ_streaming_prediction.py::test_predictions_preserve_duplicate_index_and_missing_positions[derivative-ologit]": 1.165932,
    "tests/test_editor_api_catalog.py::test_catalog_regenerates_from_committed_source_without_importing_runtime": 9.708828,
    "tests/test_editor_api_catalog.py::test_categorical_option_catalog_retains_all_metadata_and_full_logical_equality": 10.758451,
    "tests/test_editor_api_catalog.py::test_compact_generated_catalog_preserves_all_fields_without_writing": 12.322542,
    "tests/test_editor_api_catalog.py::test_complete_catalog_return_namespace_compaction_is_lossless_and_saves_payload": 10.330332,
    "tests/test_editor_api_catalog.py::test_complete_committed_catalog_parameter_compaction_stays_within_original_budget": 11.411491,
    "tests/test_editor_api_catalog.py::test_default_function_kind_encoding_preserves_complete_entry_metadata": 9.986872,
    "tests/test_editor_api_catalog.py::test_econometrics_catalog_covers_lazy_public_targets_and_exact_parameters": 7.072012,
    "tests/test_editor_api_catalog.py::test_registry_help_cannot_overwrite_a_core_export[ModelSpec]": 1.101611,
    "tests/test_editor_api_catalog.py::test_registry_help_cannot_overwrite_a_core_export[predict]": 1.029672,
    "tests/test_editor_api_catalog.py::test_return_type_tokens_preserve_full_catalog_and_save_actual_payload": 12.00268,
    "tests/test_factor_frequency_uncertainty.py::test_full_199_literal_expansion_numpy_scipy_vector_covariance_percentile_oracle[True-normal]": 1.818504,
    "tests/test_nested_logit.py::test_joint_fit_interior_and_covariance_dimensions": 2.514942,
    "tests/test_nested_logit.py::test_single_active_cases_fixed_anchor_identify_shared_beta_and_free_lambda": 1.180847,
    "tests/test_nested_logit.py::test_typed_nests_fixed_declarations_and_cases_survive_json": 1.375719,
    "tests/test_nested_logit_independent.py::test_all_physical_parameters_match_multistart_scipy[cr0]": 1.191794,
    "tests/test_nested_logit_independent.py::test_all_physical_parameters_match_multistart_scipy[hc0]": 1.170766,
    "tests/test_nested_logit_independent.py::test_all_physical_parameters_match_multistart_scipy[oim]": 1.668568,
    "tests/test_nested_logit_independent.py::test_all_prediction_companion_jacobians_and_complete_joint_covariance[cr0]": 1.075269,
    "tests/test_nested_logit_independent.py::test_all_prediction_companion_jacobians_and_complete_joint_covariance[oim]": 1.048647,
    "tests/test_nested_logit_independent.py::test_all_six_prediction_scales_complete_joint_covariance[cr0]": 1.516903,
    "tests/test_nested_logit_independent.py::test_all_six_prediction_scales_complete_joint_covariance[hc0]": 1.018581,
    "tests/test_nested_logit_independent.py::test_all_six_prediction_scales_complete_joint_covariance[oim]": 1.081378,
    "tests/test_nested_logit_independent.py::test_fixed_anchor_single_active_nests_match_independent_physical_ml_and_full_inference[cr0]": 1.490088,
    "tests/test_nested_logit_independent.py::test_fixed_anchor_single_active_nests_match_independent_physical_ml_and_full_inference[hc0]": 1.380882,
    "tests/test_nested_logit_independent.py::test_fixed_anchor_single_active_nests_match_independent_physical_ml_and_full_inference[oim]": 1.469575,
    "tests/test_nested_logit_independent.py::test_lambda_one_equals_independent_mnl": 1.063867,
    "tests/test_nested_logit_independent.py::test_missing_pair_rows_use_explicit_conditional_weight_support[cr0-elasticity]": 1.031744,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[cr0-effect]": 1.269528,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[cr0-elasticity]": 1.199944,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[hc0-effect]": 1.178542,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[hc0-elasticity]": 1.188828,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[oim-effect]": 1.143688,
    "tests/test_nested_logit_independent.py::test_own_within_cross_nest_and_unbalanced_weighted_ames[oim-elasticity]": 1.18364,
    "tests/test_nested_logit_independent.py::test_restoration_and_queries_cannot_call_optimizer[cr0]": 2.769521,
    "tests/test_nested_logit_independent.py::test_restoration_and_queries_cannot_call_optimizer[hc0]": 2.592735,
    "tests/test_nested_logit_independent.py::test_restoration_and_queries_cannot_call_optimizer[oim]": 2.595826,
    "tests/test_nested_logit_independent.py::test_row_order_preserves_parameters_after_lambda_catalogue_alignment": 1.161568,
    "tests/test_nested_logit_independent.py::test_shared_case_attribute_offsets_preserve_joint_fit": 1.131758,
    "tests/test_nonlinear_sur.py::test_joint_shared_nonlinear_ml_scores_full_information_and_profile": 4.084031,
    "tests/test_nonlinear_sur_postestimation.py::test_actual_fit_full_replay_query_restore_no_optimizer[cr0]": 1.132724,
    "tests/test_nonlinear_sur_postestimation.py::test_actual_fit_full_replay_query_restore_no_optimizer[hc0]": 1.201451,
    "tests/test_nonlinear_sur_postestimation.py::test_actual_fit_full_replay_query_restore_no_optimizer[oim]": 1.134753,
    "tests/test_parallel_sdk_groups.py::test_outer_wrapper_timeout_stops_every_child_and_grandchild": 6.641863,
    "tests/test_parallel_sdk_groups.py::test_real_total_timeout_preserves_partial_phases_and_kills_children": 1.477387,
    "tests/test_pca_frequency_uncertainty.py::test_seed_extremes_and_deterministic_complete_state[9223372036854775807]": 1.378693,
    "tests/test_repeated_gls_kernels.py::test_replay_recomputes_every_scientific_key_without_optimizer[ar1_ml]": 7.530837,
    "tests/test_repeated_gls_oracles.py::test_all_eight_external_optima_residual_geometry_and_independent_observed_information[diagonal_ml]": 1.066084,
    "tests/test_repeated_gls_oracles.py::test_all_eight_external_optima_residual_geometry_and_independent_observed_information[unstructured_ml]": 2.940476,
    "tests/test_repeated_gls_oracles.py::test_all_eight_external_optima_residual_geometry_and_independent_observed_information[unstructured_reml]": 1.937085,
    "tests/test_repeated_gls_oracles.py::test_response_and_predictor_scale_equivariance_of_complete_covariance[unstructured_ml]": 1.048522,
    "tests/test_repeated_gls_oracles.py::test_row_permutation_retains_subject_geometry_original_labels_and_inference[unstructured_ml]": 1.034458,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[ar1_ml]": 1.321288,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[ar1_reml]": 1.145938,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[cs_ml]": 1.251282,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[cs_reml]": 1.050564,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[diagonal_ml]": 1.416674,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[diagonal_reml]": 1.204507,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[unstructured_ml]": 2.017814,
    "tests/test_repeated_gls_oracles.py::test_sorted_json_restore_predictions_and_nonzero_joint_contrasts_never_optimize[unstructured_reml]": 1.65854,
    "tests/test_rm_moment_methods.py::test_rm_full_saved_joint_covariance_hotelling_t_ci_and_json[True]": 1.319927,
    "tests/test_streaming_analysis.py::test_missing_drop_retains_original_physical_positions_and_bounded_predictions": 1.671202,
    "tests/test_streaming_analysis.py::test_scan_regression_exceeds_eager_import_limit_without_full_read[csv]": 8.946861,
    "tests/test_streaming_analysis.py::test_scan_regression_exceeds_eager_import_limit_without_full_read[parquet]": 3.449665,
    "tests/test_streaming_analysis.py::test_source_digest_is_same_for_different_batch_boundaries": 1.034546,
    "tests/test_streaming_binary.py::test_final_cluster_factory_count_is_checked_against_numeric_source": 1.15539,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[1-False-logit]": 6.010322,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[1-False-probit]": 6.345204,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[1-True-logit]": 7.009731,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[1-True-probit]": 7.097519,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[7-False-probit]": 1.062953,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[7-True-logit]": 1.088145,
    "tests/test_streaming_binary.py::test_matches_dense_mle_information_and_predictions[7-True-probit]": 1.185983,
    "tests/test_streaming_binary.py::test_numeric_factory_handles_all_prior_passes_and_clusters_are_encoded_once[logit]": 1.078037,
    "tests/test_streaming_binary.py::test_numeric_factory_handles_all_prior_passes_and_clusters_are_encoded_once[probit]": 1.220692,
    "tests/test_streaming_binary.py::test_rare_classes_and_extreme_tail_rows_have_a_finite_fit[logit]": 1.80484,
    "tests/test_streaming_binary.py::test_rare_classes_and_extreme_tail_rows_have_a_finite_fit[probit]": 1.29393,
    "tests/test_streaming_control_function_engine.py::test_budgeted_original_reader_and_typed_receipts_survive_complete_replay[csv]": 8.63238,
    "tests/test_streaming_control_function_engine.py::test_budgeted_original_reader_and_typed_receipts_survive_complete_replay[parquet]": 9.266721,
    "tests/test_streaming_control_function_engine.py::test_common_missing_sample_preview_and_typed_cluster_identity": 1.248951,
    "tests/test_streaming_control_function_engine.py::test_complete_and_quasi_separation_and_fractional_interior_constraints[cloglog]": 1.366436,
    "tests/test_streaming_control_function_engine.py::test_complete_and_quasi_separation_and_fractional_interior_constraints[fractional_logit]": 1.387763,
    "tests/test_streaming_control_function_engine.py::test_complete_and_quasi_separation_and_fractional_interior_constraints[logit]": 1.554525,
    "tests/test_streaming_control_function_engine.py::test_complete_and_quasi_separation_and_fractional_interior_constraints[probit]": 1.389147,
    "tests/test_streaming_control_function_engine.py::test_each_cf_engine_entry_forces_cpu_factors_under_ambient_auto_cuda[solve]": 1.185797,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-cloglog]": 2.286738,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-fractional_logit]": 2.273836,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-gamma]": 1.722236,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-inverse_gaussian]": 2.254619,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-logit]": 2.51431,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-poisson]": 2.112567,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[None-probit]": 2.60823,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-cloglog]": 2.272209,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-fractional_logit]": 2.491725,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-gamma]": 1.773325,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-inverse_gaussian]": 2.519345,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-logit]": 2.319321,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-poisson]": 2.232191,
    "tests/test_streaming_control_function_engine.py::test_every_outcome_complete_jacobian_and_hc0_cr0_against_independent_finite_differences[cluster-probit]": 2.389118,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[cloglog]": 3.724527,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[fractional_logit]": 3.721271,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[gamma]": 2.391201,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[gaussian]": 1.59013,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[inverse_gaussian]": 3.1982,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[logit]": 3.776941,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[poisson]": 3.188847,
    "tests/test_streaming_control_function_engine.py::test_explicit_saved_parameter_replay_does_not_refit_and_is_batch_independent[probit]": 3.850969,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[cloglog]": 1.981546,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[fractional_logit]": 1.969351,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[gamma]": 1.513119,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[inverse_gaussian]": 2.200669,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[logit]": 2.028837,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[poisson]": 1.960926,
    "tests/test_streaming_control_function_engine.py::test_no_intercept_original_coordinate_factors_and_covariance[probit]": 2.108973,
    "tests/test_streaming_control_function_engine.py::test_nonconvergence_and_exhausted_separation_budget_refuse_ordinary_inference": 1.462207,
    "tests/test_streaming_control_function_engine.py::test_nullable_cluster_hash_representation_is_independent_of_missing_block[Int64]": 1.416944,
    "tests/test_streaming_control_function_engine.py::test_nullable_cluster_hash_representation_is_independent_of_missing_block[int64[pyarrow]]": 1.505489,
    "tests/test_streaming_control_function_engine.py::test_original_producer_obeys_discovery_plan_before_projection_allocation[csv]": 3.827257,
    "tests/test_streaming_control_function_engine.py::test_original_producer_obeys_discovery_plan_before_projection_allocation[parquet]": 2.883105,
    "tests/test_streaming_control_function_engine.py::test_over_5000_rows_remains_block_bounded_without_source_collection": 2.356001,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[cloglog]": 2.804908,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[fractional_logit]": 2.89471,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[gamma]": 2.119644,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[gaussian]": 1.049872,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[inverse_gaussian]": 2.837802,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[logit]": 2.890241,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[poisson]": 2.685255,
    "tests/test_streaming_control_function_engine.py::test_singleton_clusters_reduce_to_hc0_without_cr1[probit]": 3.068414,
    "tests/test_survey_four_stage_oracles.py::test_complete_example_and_standalone_native_payload_oracle": 4.336098,
    "tests/test_survey_four_stage_oracles.py::test_exact_four_real_selections_unbiased_ht_full_covariance_all_census_masks[0]": 1.20976,
    "tests/test_survey_four_stage_review.py::test_rehashed_domain_metadata_must_describe_a_possible_original_partition[None-mean]": 1.02286,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_exhaustive_three_real_stages_and_both_lower_strata_unbiased_ht_full_variance[census_stages0]": 8.679662,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_exhaustive_three_real_stages_and_both_lower_strata_unbiased_ht_full_variance[census_stages1]": 15.276668,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_exhaustive_three_real_stages_and_both_lower_strata_unbiased_ht_full_variance[census_stages2]": 1.447746,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_exhaustive_three_real_stages_and_both_lower_strata_unbiased_ht_full_variance[census_stages5]": 1.128784,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_observed_bread_weighted_likelihood_and_complete_cell_score_covariance[False-True-probit]": 1.157044,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[logit]": 1.276102,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[poisson]": 1.431067,
    "tests/test_survey_fully_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[probit]": 1.282022,
    "tests/test_survey_stratified_three_stage_oracles.py::test_exhaustive_three_stage_cell_stratified_total_has_unbiased_full_variance[census_stages1]": 1.28848,
    "tests/test_survey_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[logit]": 1.068679,
    "tests/test_survey_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[poisson]": 1.150417,
    "tests/test_survey_stratified_three_stage_oracles.py::test_physical_reordering_and_json_restore_preserve_full_joint_numerics[probit]": 1.10614,
    "tests/test_survey_stratified_three_stage_review.py::test_rehashed_saved_results_reject_frame_metadata_forgery[n_lower_strata-False-mean]": 1.365653,
    "tests/test_survey_three_stage_oracles.py::test_weighted_likelihood_observed_bread_and_three_stage_sandwich[False-True-poisson]": 1.344401,
    "tests/test_uv_script_packages.py::test_uv_failed_install_preserves_previous_manifest": 2.186668,
    "tests/test_uv_script_packages.py::test_uv_install_retains_state_and_extras_when_adding_plain_packages": 2.197917,
    "tests/test_uv_script_packages.py::test_uv_matching_range_avoids_new_job_or_worker_restart": 2.241503
}

MAX_PHASE_LINE = 8 * 1024 * 1024
MAX_PHASE_BYTES = 256 * 1024 * 1024
# The two concurrent SDK processes each own one numerical worker. Small streamed
# matrix operations must not multiply the process-level concurrency again.
GATE_ENVIRONMENT = {
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "inherited_pytest_options_removed": ["PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"],
    "explicit_plugin": "scripts.pytest_gate_timings",
}
NUMERICAL_THREAD_KEYS = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS", "ARROW_IO_THREADS")
TEMPORARY_ENVIRONMENT_KEYS = ("TMPDIR", "TMP", "TEMP")
# This stdlib bootstrap is the actual pytest process, not a separate probe.
# Resolve its temporary directory before importing pytest/application code.
SDK_CHILD_BOOTSTRAP = '''import json, os, runpy, sys, tempfile
from pathlib import Path
def require(condition, message):
    if not condition:
        raise ValueError(message)
receipt = Path(sys.argv[1])
owned = receipt.parent / "runtime-temp"
value = str(owned)
require(owned.is_absolute() and str(owned.resolve(strict=True)) == value,
        "SDK temporary root must be absolute and canonical")
require(all(not path.is_symlink() for path in (owned, *owned.parents)),
        "SDK temporary root cannot contain a symlink")
require(owned.is_dir() and not any(owned.iterdir()),
        "SDK temporary root must be a fresh empty directory")
temporary = {key: os.environ.get(key) for key in ("TMPDIR", "TMP", "TEMP")}
require(all(path == value for path in temporary.values()),
        "SDK temporary environment must belong to its exact group")
actual = tempfile.gettempdir()
require(actual == value and not any(owned.iterdir()),
        "SDK actual tempfile root differs from its fresh owned directory")
record = {"schema": 1, "pid": os.getpid(), "temporary_environment": temporary,
          "owned_directory": value, "tempfile_directory": actual,
          "canonical_directory": str(owned.resolve(strict=True)),
          "no_symlink_ancestors": True, "empty_at_bootstrap": True}
with receipt.open("x", encoding="utf-8") as output:
    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\\n")
sys.argv = ["pytest", *sys.argv[2:]]
runpy.run_module("pytest", run_name="__main__", alter_sys=True)
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selector_contract(root):
    tree = ast.parse((root / "scripts/verify_merge_candidate.py").read_text())
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "TESTS"
                      for target in node.targets)]
    require(len(values) == 1, "Missing unique source selector contract")
    selectors = values[0]
    validate_selectors(selectors)
    return selectors


def validate_selectors(selectors):
    require(isinstance(selectors, list) and len(selectors) > 1,
            "SDK selector contract must be a nonempty list")
    require(all(isinstance(item, str) and item and "\\" not in item
                and not PurePosixPath(item).is_absolute()
                and all(part not in ("", ".", "..") for part in item.split("/"))
                and "::" not in item for item in selectors), "Invalid SDK selector path")
    require(len(set(selectors)) == len(selectors), "Duplicate SDK selector")
    require(not any(a.startswith(b.rstrip("/") + "/")
                    for a in selectors for b in selectors if a != b),
            "Overlapping SDK selectors")


def split_groups(selectors, expensive=EXPENSIVE):
    validate_selectors(selectors)
    require(len(set(expensive)) == len(expensive) and set(expensive) <= set(selectors),
            "Heavy SDK selectors missing from complete source scope")
    groups = [[item for item in selectors if item in expensive],
              [item for item in selectors if item not in expensive]]
    require(all(groups), "Both SDK groups must be nonempty")
    require(Counter(item for group in groups for item in group) == Counter(selectors),
            "SDK groups must cover the original scope exactly once")
    return groups


def collection_bytes(nodes):
    require(isinstance(nodes, list) and len(nodes) >= 4, "Invalid complete partition collection")
    output = bytearray()
    for node in nodes:
        require(isinstance(node, str) and "::" in node, "Invalid partition node identity")
        file = node.split("::", 1)[0]
        require(file.endswith(".py") and "\\" not in file and not PurePosixPath(file).is_absolute()
                and all(p not in ("", ".", "..") for p in file.split("/")),
                "Invalid partition node source")
        line = (json.dumps({"nodeid": node}, allow_nan=False,
                           separators=(",", ":")) + "\n").encode()
        require(len(line) <= MAX_PHASE_LINE and len(output) + len(line) <= MAX_PHASE_BYTES,
                "Complete partition collection exceeds bounded metadata")
        output.extend(line)
    require(len(nodes) == len(set(nodes)), "Duplicate complete partition collection")
    return bytes(output)


def make_partition_plan(nodes, group_count=4):
    """Deterministic source-pinned scheduling; every collected node is assigned."""
    require(type(group_count) is int and group_count == 4,
            "Distributed SDK requires exactly four groups")
    raw = collection_bytes(nodes)
    profile = {"default_seconds": DEFAULT_NODE_SECONDS, "advisory_seconds": ADVISORY_NODE_SECONDS}
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                for x in [DEFAULT_NODE_SECONDS, *ADVISORY_NODE_SECONDS.values()]),
            "Invalid source scheduling weights")
    weights = [ADVISORY_NODE_SECONDS.get(node, DEFAULT_NODE_SECONDS) for node in nodes]
    loads, counts, assignments = [0.0] * group_count, [0] * group_count, [-1] * len(nodes)
    for index in sorted(range(len(nodes)), key=lambda i: (-weights[i], i)):
        group = min(range(group_count), key=lambda i: (loads[i], i))
        assignments[index] = group
        loads[group] += weights[index]
        counts[group] += 1
    require(all(counts), "All four SDK groups must be nonempty")
    return {"schema": 1, "kind": "source_bound_node_partition", "group_count": group_count,
            "algorithm": "stable-longest-processing-time-v1",
            "weights_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "full_collection_sha256": hashlib.sha256(raw).hexdigest(),
            "assignments": assignments, "group_counts": counts,
            "estimated_group_seconds": loads}


def partition_nodes(nodes, group_count=4):
    plan = make_partition_plan(nodes, group_count)
    return [[node for node, assigned in zip(nodes, plan["assignments"], strict=True)
             if assigned == group] for group in range(group_count)]


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def clock_identity(parent_pid=None):
    boot = Path("/proc/sys/kernel/random/boot_id")
    return {"hostname": socket.gethostname(), "pid": os.getpid() if parent_pid is None else parent_pid,
            "boot_id": boot.read_text().strip() if boot.is_file() else None}


def source_identity(root):
    return {"source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()}


def cgroup_cpu_budget(proc=Path("/proc/self")):
    """Read current membership and every visible ancestor quota, v2 or v1 CPU.

    A nested group's unlimited quota does not override its parent's CPU limit.
    Mount roots matter when the CPU hierarchy is mounted below its real root.
    Missing or malformed quota evidence yields a conservative one-CPU budget.
    """
    result = {"status": "unavailable", "version": None, "views": [], "probes": [],
              "limits": [], "error": None}
    try:
        members = [line.split(":", 2) for line in (proc / "cgroup").read_text().splitlines()]
        require(all(len(item) == 3 for item in members), "Malformed cgroup membership")
        mounts = []
        for line in (proc / "mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fields, kind = left.split(), right.split()
            require(len(fields) >= 6 and len(kind) >= 3, "Malformed cgroup mount metadata")
            if kind[0] == "cgroup2" or (kind[0] == "cgroup" and "cpu" in kind[2].split(",")):
                def unescape(value):
                    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
                mounts.append((2 if kind[0] == "cgroup2" else 1,
                               PurePosixPath(unescape(fields[3])), Path(unescape(fields[4]))))
        # Prefer the unified hierarchy when it actually exposes the CPU controller.
        for version in (2, 1):
            result.update(version=None, views=[], probes=[], limits=[])
            for candidate, mount_root, mount in mounts:
                if candidate != version:
                    continue
                matching = [PurePosixPath(path) for _, controllers, path in members
                            if (version == 2 and controllers == "")
                            or (version == 1 and "cpu" in controllers.split(","))]
                for member in matching:
                    require(member.is_absolute() and ".." not in member.parts
                            and mount_root.is_absolute() and ".." not in mount_root.parts
                            and mount.is_absolute() and ".." not in mount.parts,
                            "Malformed cgroup path")
                    if not member.is_relative_to(mount_root):
                        continue
                    result["version"] = version
                    view = len(result["views"])
                    result["views"].append({"membership": str(member), "mount_root": str(mount_root),
                                            "mountpoint": str(mount)})
                    current = mount.joinpath(*member.relative_to(mount_root).parts)
                    while True:
                        path = current / ("cpu.max" if version == 2 else "cpu.cfs_quota_us")
                        present = path.exists()
                        result["probes"].append({"view": view, "path": str(path),
                                                 "status": "present" if present else "missing"})
                        if present:
                            if version == 2:
                                values = path.read_text().split()
                                require(len(values) == 2, "Malformed cgroup v2 CPU quota")
                                quota = None if values[0] == "max" else int(values[0])
                                period = int(values[1])
                            else:
                                quota = int(path.read_text().strip())
                                quota = None if quota == -1 else quota
                                period = int((current / "cpu.cfs_period_us").read_text().strip())
                            require(period > 0 and (quota is None or quota > 0),
                                    "Malformed cgroup CPU quota or period")
                            cpus = None if quota is None else quota / period
                            require(cpus is None or (math.isfinite(cpus) and cpus > 0),
                                    "Nonfinite cgroup CPU quota")
                            result["limits"].append({"view": view, "path": str(path), "quota_us": quota,
                                                     "period_us": period, "cpus": cpus})
                        if current == mount:
                            break
                        current = current.parent
            # Examine every applicable view before deciding this hierarchy's limit.
            if result["limits"]:
                result["status"] = ("limited" if any(item["cpus"] is not None
                                                    for item in result["limits"]) else "unlimited")
                return result
        result["error"] = "No readable quota in the current visible CPU hierarchy"
    except (OSError, ValueError, OverflowError) as error:
        result.update(status="unavailable" if isinstance(error, OSError) else "malformed",
                      error=f"{type(error).__name__}: {error}")
    return result


def detect_cpu_budget(*, proc=Path("/proc/self"), system=None):
    system = platform.system() if system is None else system
    try:
        host = os.cpu_count()
    except (OSError, NotImplementedError):
        host = None
    host = host if type(host) is int and host > 0 else None
    try:
        affinity = len(os.sched_getaffinity(0))
        affinity = affinity if affinity > 0 else None
    except (AttributeError, OSError, NotImplementedError):
        affinity = None
    cgroup = (cgroup_cpu_budget(proc) if system == "Linux" else
              {"status": "not_applicable", "version": None, "views": [], "probes": [],
               "limits": [], "error": None})
    limits = [value for value in (host, affinity) if value is not None]
    limits.extend(item["cpus"] for item in cgroup["limits"] if item["cpus"] is not None)
    if cgroup["status"] in ("unavailable", "malformed"):
        limits.append(1)
    effective = max(1.0, min(limits, default=1.0))
    return {"system": system, "host_cpus": host, "affinity_cpus": affinity, "cgroup": cgroup,
            "effective_cpus": effective, "groups": 2,
            "max_concurrent_children": min(2, max(1, math.floor(effective))),
            "threads_per_child": min(2, max(1, math.floor(effective / 2)))}


def child_environment_contract(threads):
    require(type(threads) is int and threads in (1, 2), "Invalid SDK child thread allocation")
    return {**GATE_ENVIRONMENT, **{key: str(threads) for key in NUMERICAL_THREAD_KEYS}}


def environment(root, inherited=None, *, threads=2, distributed=False, temporary=None):
    threads = 1 if distributed else threads
    child_environment_contract(threads)
    env = dict(os.environ if inherited is None else inherited)
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        env.pop(key, None)
    env.update(PYTHONPATH=os.pathsep.join([str(root / "src"),
                                         str(root / "packages/openecon-charts/src")]),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               **{key: str(threads) for key in NUMERICAL_THREAD_KEYS})
    if temporary is not None:
        owned_temporary_directory(temporary)
        env.update({key: str(temporary) for key in TEMPORARY_ENVIRONMENT_KEYS})
    return env


def owned_temporary_directory(path):
    require(isinstance(path, Path) and path.is_absolute()
            and ".." not in path.parts and path.resolve() == path,
            "SDK temporary directory must be an absolute canonical owned path")
    require(all(not item.is_symlink() for item in (path, *path.parents)),
            "SDK temporary directory cannot inherit a symlink")
    require(path.is_dir(), "SDK temporary directory is missing or is not a directory")


def child_temporary_receipt(path, temporary, pid, *, removed=False):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4096,
            "Missing bounded actual SDK child temporary environment receipt")
    if removed:
        require(temporary.is_absolute() and ".." not in temporary.parts
                and temporary.resolve() == temporary
                and all(not item.is_symlink() for item in (temporary, *temporary.parents))
                and not temporary.exists(), "SDK owned temporary directory was not removed")
    else:
        owned_temporary_directory(temporary)
    record = _json(path.read_bytes())
    expected = {"schema": 1, "pid": pid,
                "temporary_environment": {key: str(temporary) for key in TEMPORARY_ENVIRONMENT_KEYS},
                "owned_directory": str(temporary), "tempfile_directory": str(temporary),
                "canonical_directory": str(temporary), "no_symlink_ancestors": True,
                "empty_at_bootstrap": True}
    require(record == expected and type(record.get("schema")) is int
            and type(record.get("pid")) is int
            and record.get("no_symlink_ancestors") is True
            and record.get("empty_at_bootstrap") is True,
            "Actual SDK child temporary directory/environment differs from its owned group")
    return record


def remove_child_temporary_directory(path, identity):
    owned_temporary_directory(path)
    metadata = path.stat()
    require(identity == (metadata.st_dev, metadata.st_ino)
            and shutil.rmtree.avoids_symlink_attacks,
            "SDK temporary cleanup refused a replaced directory or unsafe remover")
    # The fd-based remover does not follow links inside this originally owned root.
    shutil.rmtree(path)
    require(not path.exists() and not path.is_symlink(), "SDK owned temporary cleanup failed")


def child_command(python, root, directory, selectors, *, group_index=None):
    command = [str(python), "-c", SDK_CHILD_BOOTSTRAP,
            str(directory / "runtime-environment.json"), "-c", str(root / "pyproject.toml"),
            "-p", "scripts.pytest_gate_timings", "--gate-timings",
            str(directory / "pytest-timings.jsonl"), "--gate-collection",
            str(directory / "pytest-collection.jsonl")]
    if group_index is not None:
        require(type(group_index) is int and 0 <= group_index < 4, "Invalid global SDK group")
        command += ["--gate-full-collection", str(directory / "pytest-full-collection.jsonl"),
                    "--gate-partition-plan", str(directory / "partition-plan.json"),
                    "--gate-group-index", str(group_index), "--gate-group-count", "4"]
    return [*command, "--junitxml", str(directory / "pytest.xml"),
            "--basetemp", str(directory / "pytest-temp"),
            "-o", "cache_dir=" + str(directory / "pytest-cache"), *selectors]


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate phase metadata key")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("Nonfinite phase metadata")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def selector_index(file, selectors):
    matches = [index for index, selector in enumerate(selectors)
               if file == selector or file.startswith(selector.rstrip("/") + "/")]
    require(len(matches) == 1, "Phase source must bind to exactly one selected path")
    return matches[0]


def collected_nodes(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_PHASE_BYTES,
            "Missing or excessive same-process collection evidence")
    nodes = []
    with path.open("rb") as stream:
        for line in iter(lambda: stream.readline(MAX_PHASE_LINE + 1), b""):
            require(len(line) <= MAX_PHASE_LINE and line.strip(), "Invalid collection record")
            item = _json(line)
            require(isinstance(item, dict) and set(item) == {"nodeid"}
                    and isinstance(item["nodeid"], str) and "::" in item["nodeid"],
                    "Invalid collected node identity")
            nodes.append(item["nodeid"])
    require(nodes and len(nodes) == len(set(nodes)), "Empty or duplicate collection evidence")
    return nodes


def read_child(junit, timings, selectors, *, expected_nodes=None):
    require(junit.is_file() and not junit.is_symlink()
            and junit.stat().st_size <= 32 * 1024 * 1024, "Missing or excessive child JUnit")
    raw = junit.read_bytes()
    require(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw,
            "JUnit declarations are not accepted")
    tree = ET.fromstring(raw)
    require(tree.tag in ("testsuite", "testsuites"), "Invalid child JUnit")
    require(not any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")),
            "Child JUnit is failed, errored or skipped")
    suites = [suite for suite in tree.iter("testsuite") if not suite.findall("testsuite")]
    cases, duration = [], 0.0
    for suite in suites:
        attrs = {}
        for key in ("tests", "failures", "errors", "skipped"):
            value = suite.get(key)
            require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value),
                    "Invalid child JUnit counts")
            attrs[key] = int(value)
        actual = suite.findall("testcase")
        require(attrs["tests"] == len(actual) and actual
                and not any(attrs[key] for key in ("failures", "errors", "skipped")),
                "Child JUnit must contain actual nonempty passing cases")
        seconds = float(suite.get("time", "0"))
        require(math.isfinite(seconds) and seconds >= 0, "Invalid child JUnit duration")
        duration += seconds
        cases.extend(actual)
    require(cases and len(list(tree.iter("testcase"))) == len(cases),
            "Missing or uncounted child JUnit cases")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    require(all(isinstance(a, str) and a and isinstance(b, str) and b for a, b in identities)
            and len(set(identities)) == len(identities), "Duplicate or invalid child JUnit identity")
    require(timings.is_file() and not timings.is_symlink()
            and timings.stat().st_size <= MAX_PHASE_BYTES, "Missing or excessive child phases")
    nodes, triples, pending = [], [], []
    with timings.open("rb") as stream:
        for line in iter(lambda: stream.readline(MAX_PHASE_LINE + 1), b""):
            require(len(line) <= MAX_PHASE_LINE and line.strip(), "Invalid child phase record")
            record = _json(line)
            require(isinstance(record, dict)
                    and set(record) == {"nodeid", "phase", "outcome", "duration", "start", "stop"},
                    "Invalid child phase fields")
            require(record["outcome"] == "passed", "Child phase is not passed")
            require(all(type(record[key]) in (int, float) and math.isfinite(record[key])
                        and record[key] >= 0 for key in ("duration", "start", "stop"))
                    and record["stop"] >= record["start"], "Invalid child phase timing")
            node = record["nodeid"]
            require(isinstance(node, str) and "::" in node, "Invalid child phase identity")
            file = node.split("::", 1)[0]
            require(file.endswith(".py") and not PurePosixPath(file).is_absolute()
                    and ".." not in PurePosixPath(file).parts, "Invalid child phase source")
            selector_index(file, selectors)
            require(record["phase"] == ("setup", "call", "teardown")[len(pending)]
                    and (not pending or pending[0][0] == node), "Incomplete or reordered child phases")
            pending.append((node, line))
            if len(pending) == 3:
                nodes.append(node)
                triples.append(tuple(item[1] for item in pending))
                pending = []
    require(not pending and len(nodes) == len(cases), "Incomplete child phase evidence")
    require(len(set(nodes)) == len(nodes), "Duplicate child phase testcase")
    files = {node.split("::", 1)[0] for node in nodes}
    mapped = []
    for classname, name in identities:
        matches = []
        for file in files:
            module = file[:-3].replace("/", ".")
            if classname == module:
                matches.append(file + "::" + name)
            elif classname.startswith(module + "."):
                matches.append(file + "::" + classname[len(module) + 1:].replace(".", "::")
                               + "::" + name)
        require(len(matches) == 1, "Child JUnit cannot bind to one phase identity")
        mapped.append(matches[0])
    require(mapped == nodes, "Child JUnit/phase case order or identities differ")
    require(collected_nodes(junit.parent / "pytest-collection.jsonl") == nodes,
            "Raw JUnit/phases omit or reorder same-process collected tests")
    if expected_nodes is not None:
        require(nodes == expected_nodes, "Child execution differs from its complete derived partition")
    ranks = [selector_index(node.split("::", 1)[0], selectors) for node in nodes]
    require(ranks == sorted(ranks) and (expected_nodes is not None
            or set(ranks) == set(range(len(selectors)))),
            "Child source coverage missing or reordered")
    return list(zip(nodes, cases, triples)), duration


def merge_evidence(children, selectors, groups, junit, timings):
    require(groups == split_groups(selectors, tuple(groups[0])),
            "SDK child groups differ from their disjoint ordered partition")
    rows, duration = [], 0.0
    for child, group in zip(children, groups, strict=True):
        actual, seconds = read_child(child / "pytest.xml", child / "pytest-timings.jsonl", group)
        rows.extend(actual)
        duration += seconds
    require(len({row[0] for row in rows}) == len(rows), "Duplicate testcase across SDK groups")
    rows.sort(key=lambda row: selector_index(row[0].split("::", 1)[0], selectors))
    counts = {"tests": len(rows), "failures": 0, "errors": 0, "skipped": 0}
    suite = ET.Element("testsuite", name="pytest-sdk-groups", time=str(duration),
                       **{key: str(value) for key, value in counts.items()})
    for _, case, _ in rows:
        suite.append(case)
    root = ET.Element("testsuites")
    root.append(suite)
    with timings.open("wb") as output:
        for _, _, records in rows:
            output.writelines(records)
    ET.ElementTree(root).write(junit, encoding="utf-8", xml_declaration=True)
    return counts


def merge_partition_evidence(children, selectors, group_indices, junit, timings):
    """Validate a shard's actual partial rows; aggregate alone proves all four."""
    validate_selectors(selectors)
    require(len(children) == len(group_indices) == 2
            and group_indices in ([0, 1], [2, 3]), "Invalid SDK shard child indexes")
    full, rows, duration = None, [], 0.0
    for child, group in zip(children, group_indices, strict=True):
        nodes = collected_nodes(child / "pytest-full-collection.jsonl")
        require((child / "pytest-full-collection.jsonl").read_bytes() == collection_bytes(nodes),
                "Full collection bytes differ from canonical source collection")
        ranks = [selector_index(node.split("::", 1)[0], selectors) for node in nodes]
        require(ranks == sorted(ranks) and set(ranks) == set(range(len(selectors))),
                "Full source selector collection missing or reordered")
        require(full is None or nodes == full, "Shard children collected different complete SDK scope")
        full = nodes
        path = child / "partition-plan.json"
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024,
                "Missing or excessive source-bound partition plan")
        plan = _json(path.read_bytes())
        expected = make_partition_plan(nodes)
        require(plan == expected, "Partition plan differs from source-bound complete collection")
        selected = [node for node, index in zip(nodes, expected["assignments"], strict=True)
                    if index == group]
        actual, seconds = read_child(child / "pytest.xml", child / "pytest-timings.jsonl",
                                     selectors, expected_nodes=selected)
        rows.extend(actual)
        duration += seconds
    require(len({row[0] for row in rows}) == len(rows), "Duplicate testcase across shard children")
    order = {node: index for index, node in enumerate(full)}
    rows.sort(key=lambda row: order[row[0]])
    counts = {"tests": len(rows), "failures": 0, "errors": 0, "skipped": 0}
    suite = ET.Element("testsuite", name="pytest-sdk-shard", time=str(duration),
                       **{key: str(value) for key, value in counts.items()})
    for _, case, _ in rows:
        suite.append(case)
    tree = ET.Element("testsuites")
    tree.append(suite)
    with timings.open("wb") as output:
        for _, _, records in rows:
            output.writelines(records)
    ET.ElementTree(tree).write(junit, encoding="utf-8", xml_declaration=True)
    return counts


def _stop(processes):
    # Signal process groups even if their pytest leader has already exited.
    for process in processes:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 2
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_groups(*, root, python, selectors, directory, junit, timings, timeout=1800,
               expensive=EXPENSIVE, env=None, execution=None, component=None,
               parent_started=None, parent_deadline=None, execution_sha256=None,
               shard_index=None, shard_count=2, parent_started_utc=None, parent_pid=None):
    started = time.monotonic() if parent_started is None else parent_started
    require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
            "Invalid SDK total deadline")
    require(type(started) in (int, float) and math.isfinite(started)
            and 0 <= started <= time.monotonic(), "Invalid parent SDK clock")
    # One CI scheduling budget covers both children and verification, not scientific admission.
    deadline = started + timeout
    require(parent_deadline is None or parent_deadline == deadline,
            "Parent SDK deadline differs from original total budget")
    distributed = shard_index is not None
    require(not distributed or (type(shard_index) is int and shard_index in (0, 1)
            and type(shard_count) is int and shard_count == 2), "Invalid SDK shard0..1/2")
    if distributed:
        validate_selectors(selectors)
        groups = [selectors, selectors]
        indices = [shard_index * 2, shard_index * 2 + 1]
        started_utc = time.time() - (time.monotonic() - started) if parent_started_utc is None else parent_started_utc
        require(type(started_utc) in (int, float) and math.isfinite(started_utc)
                and started_utc > 0 and abs((time.time() - started_utc)
                - (time.monotonic() - started)) <= 1, "Parent SDK UTC/monotonic clocks differ")
        require(parent_pid is None or type(parent_pid) is int and parent_pid > 0,
                "Invalid SDK parent process identity")
    else:
        groups = split_groups(selectors, expensive)
        indices = list(range(2))
    cpu_budget = detect_cpu_budget()
    if distributed:
        cpu_budget["threads_per_child"] = 1
    child_environment = child_environment_contract(cpu_budget["threads_per_child"])
    require(directory.is_absolute() and ".." not in directory.parts
            and directory.resolve() == directory
            and all(not path.is_symlink() for path in (directory, *directory.parents)),
            "SDK group directory must be a fresh canonical non-symlink owned path")
    directory.mkdir(parents=True, exist_ok=False)
    require(not any(path.exists() or path.is_symlink() for path in (junit, timings)),
            "SDK merged outputs must be fresh")
    binding = source_identity(root)
    if execution is not None:
        require(execution.get("source_commit") == binding["source_commit"]
                and execution.get("component_sdk_version") == component
                and component in ("3.11", "3.13"), "SDK execution/source binding differs")
        require(not distributed or (type(execution.get("component_sdk_shard")) is int
                and execution["component_sdk_shard"] == shard_index),
                "SDK shard execution binding differs")
    report = {"schema": 1, "kind": "parallel_sdk_groups", "status": "running", **binding,
              "selected_tests": selectors, "timeout_seconds": timeout, "groups": [],
              "checkout_root": str(root), "python": str(python),
              "component_sdk_version": component, "execution": execution,
              "execution_sha256": execution_sha256,
              "environment": child_environment, "cpu_budget": cpu_budget,
              "duration_precision": "seconds rounded to six decimal places; one parent monotonic clock",
              "ordering": "Raw children preserve execution order; merged actual cases follow original selectors"}
    if distributed:
        report.update(schema=2, kind="distributed_sdk_shard", shard_index=shard_index,
                      shard_count=2, group_count=4, partial_scope=True,
                      job_name=f"OpenEconometrics / SDK Python {component} shard {shard_index}",
                      ordering="Raw child selected IDs preserve full collection order; partial merged cases follow complete source collection")
    processes, logs, temporary_owners = [], [], {}
    old_handlers = {}
    interruption = None

    def interrupted(signum, _frame):
        # Do not raise between Popen returning and registering its process group.
        nonlocal interruption
        interruption = signum

    def remaining():
        if interruption is not None:
            raise InterruptedError(f"SDK grouping interrupted by signal {interruption}")
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("SDK total wall deadline exceeded")
        return value

    def save():
        finished = time.monotonic()
        report["seconds"] = round(finished - started, 6)
        if distributed:
            finished_utc = time.time()
            report["sdk_clock"] = {"started_utc": started_utc, "finished_utc": finished_utc,
                                   "elapsed_seconds": finished - started,
                                   "monotonic_started": started, "monotonic_finished": finished,
                                   "clock_identity": clock_identity(parent_pid)}
            require(abs((finished_utc - started_utc) - (finished - started)) <= 1,
                    "SDK UTC clock rolled back or diverged from monotonic elapsed")
        (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupted)
        for index, group in zip(indices, groups, strict=True):
            remaining()
            while sum(process.poll() is None for process in processes) >= cpu_budget["max_concurrent_children"]:
                remaining()
                require(not any(process.poll() not in (None, 0) for process in processes),
                        "SDK child process failed")
                for item, previous in zip(report["groups"], processes, strict=True):
                    if previous.poll() is not None and "stop_seconds" not in item:
                        item["stop_seconds"] = round(time.monotonic() - started, 6)
                        if distributed:
                            item["stop_utc"] = time.time()
                time.sleep(min(0.05, remaining()))
            require(not any(process.poll() not in (None, 0) for process in processes),
                    "SDK child process failed")
            for item, previous in zip(report["groups"], processes, strict=True):
                if previous.poll() is not None and "stop_seconds" not in item:
                    item["stop_seconds"] = round(time.monotonic() - started, 6)
                    if distributed:
                        item["stop_utc"] = time.time()
            child = directory / f"group-{index}"
            child.mkdir()
            temporary = child / "runtime-temp"
            require(not temporary.exists() and not temporary.is_symlink(),
                    "SDK child temporary directory must be newly absent")
            temporary.mkdir(mode=0o700, exist_ok=False)
            owned_temporary_directory(temporary)
            metadata = temporary.stat()
            temporary_owners[index] = (temporary, (metadata.st_dev, metadata.st_ino))
            command = child_command(python, root, child, group,
                                    group_index=index if distributed else None)
            log = (child / "pytest.log").open("w")
            logs.append(log)
            child_started = time.monotonic() - started
            process = subprocess.Popen(command, cwd=root,
                                       env=environment(root, env, threads=cpu_budget["threads_per_child"],
                                                       temporary=temporary),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt")
            processes.append(process)
            report["groups"].append({"index": index, "selectors": group, "command": command,
                                     "status": "running", "pid": process.pid,
                                     "start_seconds": round(child_started, 6),
                                     "environment": {**child_environment, **{
                                         key: str(temporary) for key in TEMPORARY_ENVIRONMENT_KEYS}},
                                     "temporary_directory": str(temporary),
                                     "temporary_directory_fresh_before_launch": True,
                                     "execution": execution})
            if distributed:
                report["groups"][-1]["start_utc"] = time.time()
            save()
        save()
        while any(process.poll() is None for process in processes):
            remaining()
            for item, process in zip(report["groups"], processes, strict=True):
                if process.poll() is not None and "stop_seconds" not in item:
                    item["stop_seconds"] = round(time.monotonic() - started, 6)
                    if distributed:
                        item["stop_utc"] = time.time()
            require(not any(process.poll() not in (None, 0) for process in processes),
                    "SDK child process failed")
            time.sleep(min(0.05, remaining()))
        require(all(process.returncode == 0 for process in processes), "SDK child process failed")
        for item in report["groups"]:
            item.setdefault("stop_seconds", round(time.monotonic() - started, 6))
            if distributed:
                item.setdefault("stop_utc", time.time())
            child = directory / f"group-{item['index']}"
            child_temporary_receipt(child / "runtime-environment.json",
                                    child / "runtime-temp", item["pid"])
        remaining()
        children = [directory / f"group-{i}" for i in indices]
        counts = (merge_partition_evidence(children, selectors, indices, junit, timings)
                  if distributed else merge_evidence(children, selectors, groups, junit, timings))
        require(source_identity(root) == binding, "SDK source changed during execution")
        remaining()
        report.update(status="passed", tests=counts, junit_sha256=digest(junit),
                      phase_timings_sha256=digest(timings),
                      combined={"junit": str(junit.relative_to(directory.parent)),
                                "junit_sha256": digest(junit),
                                "phase_timings": str(timings.relative_to(directory.parent)),
                                "phase_timings_sha256": digest(timings), "tests": counts})
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        _stop(processes)
    finally:
        _stop(processes)
        for log in logs:
            log.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        for index, (temporary, identity) in temporary_owners.items():
            item = next((group for group in report["groups"] if group["index"] == index), None)
            try:
                remove_child_temporary_directory(temporary, identity)
                if item is not None:
                    item["temporary_directory_removed_after_stop"] = True
            except (OSError, ValueError) as error:
                if item is not None:
                    item["temporary_directory_removed_after_stop"] = False
                    item["temporary_cleanup_error"] = f"{type(error).__name__}: {error}"
                report.update(status="failed", error=f"SDK owned temporary cleanup failed: {type(error).__name__}: {error}")
        for item, process in zip(report["groups"], processes, strict=True):
            item.setdefault("stop_seconds", round(time.monotonic() - started, 6))
            if distributed:
                item.setdefault("stop_utc", time.time())
            item["seconds"] = round(item["stop_seconds"] - item["start_seconds"], 6)
            item["exit_code"] = process.returncode
            item["status"] = "passed" if process.returncode == 0 else "failed"
            child = directory / f"group-{item['index']}"
            for key, name in (("log", "pytest.log"), ("junit", "pytest.xml"),
                              ("phase_timings", "pytest-timings.jsonl"),
                              ("collection", "pytest-collection.jsonl"),
                              ("runtime_environment", "runtime-environment.json")):
                path = child / name
                if path.is_file():
                    item[key] = str(path.relative_to(directory))
                    item[key + "_sha256"] = digest(path)
            if distributed:
                for key, name in (("full_collection", "pytest-full-collection.jsonl"),
                                  ("partition_plan", "partition-plan.json")):
                    path = child / name
                    if path.is_file():
                        item[key] = str(path.relative_to(directory))
                        item[key + "_sha256"] = digest(path)
            if report["status"] == "passed":
                expected_nodes = None
                if distributed:
                    full = collected_nodes(child / "pytest-full-collection.jsonl")
                    expected_nodes = partition_nodes(full)[item["index"]]
                    item["full_collected_tests"] = len(full)
                    item["partition_plan_digest"] = plan_digest(make_partition_plan(full))
                rows, _ = read_child(child / "pytest.xml", child / "pytest-timings.jsonl",
                                     item["selectors"], expected_nodes=expected_nodes)
                item["tests"] = {"tests": len(rows), "failures": 0, "errors": 0, "skipped": 0}
                item["collected_tests"] = len(rows)
        if report["status"] == "passed":
            try:
                remaining()
            except (TimeoutError, InterruptedError) as error:
                report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--junitxml", type=Path, required=True)
    parser.add_argument("--gate-timings", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=None, choices=[900, 1800])
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--component-sdk-version", choices=["3.11", "3.13"], required=True)
    parser.add_argument("--shard-index", type=int, choices=[0, 1])
    parser.add_argument("--shard-count", type=int, choices=[2], default=2)
    args = parser.parse_args()
    # The distributed hosted cohort and legacy coordinator have distinct CI allocations.
    required_timeout = 900 if args.shard_index is not None else 1800
    if args.timeout is None:
        args.timeout = required_timeout
    elif args.timeout != required_timeout:
        parser.error(f"This SDK mode retains its {required_timeout}-second deadline")
    require(args.execution.is_file() and not args.execution.is_symlink()
            and args.execution.stat().st_size <= 2 * 1024 * 1024, "Missing SDK execution receipt")
    execution = _json(args.execution.read_bytes())
    report = run_groups(root=ROOT, python=args.python.absolute(), selectors=selector_contract(ROOT),
                        directory=args.directory.absolute(), junit=args.junitxml.absolute(),
                        timings=args.gate_timings.absolute(), timeout=args.timeout,
                        execution=execution, component=args.component_sdk_version,
                        execution_sha256=digest(args.execution),
                        parent_started=float(os.environ["OPENECON_SDK_PARENT_STARTED_MONOTONIC"]),
                        parent_deadline=float(os.environ["OPENECON_SDK_PARENT_DEADLINE_MONOTONIC"]),
                        shard_index=args.shard_index, shard_count=args.shard_count,
                        parent_started_utc=float(os.environ["OPENECON_SDK_PARENT_STARTED_UTC"])
                            if args.shard_index is not None else None,
                        parent_pid=int(os.environ["OPENECON_SDK_PARENT_PID"])
                            if args.shard_index is not None else None)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"],
                      "report": str(args.directory / "report.json"), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
