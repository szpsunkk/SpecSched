.PHONY: motivation-smoke motivation-run motivation-plot motivation-clean
.PHONY: real-paper real-paper-bg real-paper-dry-run real-paper-process real-paper-figures real-paper-smoke real-paper-poisson-smoke real-paper-bursty-smoke real-paper-multitier-smoke real-paper-transfer-smoke real-paper-transfer-gpu real-paper-transfer-gpu-process real-paper-claim-audit real-paper-tpds-gap-audit real-paper-focused-audits tpds-results-reproduce tpds-artifact-check tpds-artifact-package tpds-artifact-package-check tpds-readiness tpds-anonymous tpds-submission-check tpds-package tpds-package-check tpds-readiness-report tpds-anonymous-package tpds-anonymous-package-check tpds-final-preflight tpds-upload-check tpds-upload-summary
.PHONY: real-paper-tpds-multiseed real-paper-tpds-multiseed-bg real-paper-tpds-multiseed-dry-run real-paper-tpds-multiseed-process real-paper-tpds-multiseed-process-partial real-paper-tpds-multiseed-status real-paper-tpds-progress-report real-paper-heterog-poisson real-paper-heterog-poisson-dry-run real-paper-heterog-poisson-process real-paper-heterog-poisson-multiseed real-paper-heterog-poisson-multiseed-bg real-paper-heterog-poisson-multiseed-dry-run real-paper-heterog-poisson-multiseed-process real-paper-heterog-poisson-multiseed-process-partial real-paper-heterog-poisson-multiseed-status real-paper-supplement-multiseed real-paper-supplement-multiseed-bg real-paper-supplement-multiseed-dry-run real-paper-supplement-multiseed-process real-paper-supplement-multiseed-status

motivation-smoke:
	bash experiments/motivation/scripts/run_smoke.sh

motivation-run:
	bash experiments/motivation/scripts/run_edge_benchmark.sh

motivation-plot:
	bash experiments/motivation/scripts/plot.sh

motivation-clean:
	rm -f experiments/motivation/results/serving_smoke*.csv
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# Main paper sweep: local 7B target + local 0.5B draft, adaptive bucket SLO,
# N=4..64, KV budget sweep 25/50/75/100%, beta sensitivity. See ex.md.
real-paper:
	bash experiments/real_matrix/scripts/run_paper.sh

real-paper-bg:
	bash experiments/real_matrix/scripts/run_paper_background.sh

real-paper-dry-run:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/paper3090.json} --dry-run

real-paper-process:
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/paper3090.json}

real-paper-figures:
	python experiments/real_matrix/plot_configured_paper.py --config $${CONFIG:-configs/paper3090.json}

real-paper-smoke:
	python experiments/real_matrix/smoke_test.py --config $${CONFIG:-configs/smoke_paper.json}

real-paper-poisson-smoke:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/smoke_poisson_paper.json}
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/smoke_poisson_paper.json}

real-paper-bursty-smoke:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/smoke_bursty_paper.json}
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/smoke_bursty_paper.json}

real-paper-multitier-smoke:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/smoke_multitier_paper.json}
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/smoke_multitier_paper.json}

real-paper-trace-smoke:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/smoke_trace_replay_paper.json}
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/smoke_trace_replay_paper.json}

real-paper-transfer-smoke:
	python experiments/real_matrix/generate_transfer_traces.py --smoke --seeds 123 --out-dir "$${PROJECT_ROOT:-$(CURDIR)}/data/transfer_traces_smoke"
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/smoke_transfer_gpu.json}
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/smoke_transfer_gpu.json}
	python experiments/real_matrix/process_transfer_gpu_results.py --config $${CONFIG:-configs/smoke_transfer_gpu.json}

real-paper-transfer-gpu:
	python experiments/real_matrix/generate_transfer_traces.py --seeds 7 --out-dir "$${PROJECT_ROOT:-$(CURDIR)}/data/transfer_traces"
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/paper3090_transfer_gpu.json}
	$(MAKE) real-paper-transfer-gpu-process CONFIG=$${CONFIG:-configs/paper3090_transfer_gpu.json}

real-paper-transfer-gpu-process:
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/paper3090_transfer_gpu.json}
	python experiments/real_matrix/process_transfer_gpu_results.py --config $${CONFIG:-configs/paper3090_transfer_gpu.json}

real-paper-claim-audit:
	python scripts/audit_paper_claims.py

real-paper-tpds-gap-audit:
	python scripts/audit_tpds_gaps.py

real-paper-focused-audits:
	python experiments/scheduler/scripts/verify_latency_perturbation_audit.py
	python experiments/scheduler/scripts/estimator_posalpha_audit.py
	python scripts/audit_tpds_gaps.py

tpds-results-reproduce:
	bash scripts/reproduce_tpds_results.sh

real-paper-tpds-progress-report:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json} --allow-partial
	$(MAKE) real-paper-focused-audits
	python scripts/write_tpds_progress_report.py

tpds-artifact-check:
	python scripts/verify_tpds_artifact.py

tpds-artifact-package:
	python scripts/package_tpds_artifact.py

tpds-artifact-package-check:
	python scripts/verify_tpds_artifact_package.py

tpds-readiness:
	python scripts/check_tpds_readiness.py

tpds-anonymous:
	python scripts/build_anonymous_manuscript.py

tpds-submission-check: tpds-readiness tpds-anonymous

tpds-package:
	python scripts/package_tpds_submission.py

tpds-package-check:
	python scripts/verify_tpds_package.py

tpds-readiness-report:
	python scripts/write_tpds_readiness_report.py

tpds-anonymous-package:
	python scripts/package_tpds_anonymous_submission.py

tpds-anonymous-package-check:
	python scripts/verify_tpds_anonymous_package.py

tpds-final-preflight: tpds-submission-check tpds-package tpds-anonymous-package tpds-artifact-package tpds-package-check tpds-anonymous-package-check tpds-artifact-package-check tpds-readiness-report
	sha256sum dist/tpds_submission.zip dist/tpds_anonymous_submission.zip dist/tpds_artifact.zip
	stat -c '%n %s' dist/tpds_submission.zip dist/tpds_anonymous_submission.zip dist/tpds_artifact.zip

tpds-upload-check: tpds-readiness tpds-package-check tpds-anonymous-package-check tpds-artifact-package-check
	python scripts/verify_tpds_records.py
	sha256sum dist/tpds_submission.zip dist/tpds_anonymous_submission.zip dist/tpds_artifact.zip
	stat -c '%n %s' dist/tpds_submission.zip dist/tpds_anonymous_submission.zip dist/tpds_artifact.zip

tpds-upload-summary:
	python scripts/tpds_upload_summary.py

real-paper-tpds-multiseed:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_main.json}

real-paper-tpds-multiseed-bg:
	bash experiments/real_matrix/scripts/run_tpds_multiseed_background.sh

real-paper-tpds-multiseed-dry-run:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_main.json} --dry-run

real-paper-tpds-multiseed-process:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_main.json}

real-paper-tpds-multiseed-process-partial:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_main.json} --allow-partial

real-paper-tpds-multiseed-status:
	python experiments/real_matrix/status_multiseed.py --config $${CONFIG:-configs/tpds_multiseed_main.json}

real-paper-heterog-poisson:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/paper3090_heterog_poisson.json}

real-paper-heterog-poisson-dry-run:
	python experiments/real_matrix/run_configured_paper.py --config $${CONFIG:-configs/paper3090_heterog_poisson.json} --dry-run

real-paper-heterog-poisson-process:
	python experiments/real_matrix/process_configured_results.py --config $${CONFIG:-configs/paper3090_heterog_poisson.json}

real-paper-heterog-poisson-multiseed:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json}

real-paper-heterog-poisson-multiseed-bg:
	CONFIG=$${CONFIG:-configs/tpds_multiseed_heterog_poisson.json} bash experiments/real_matrix/scripts/run_tpds_multiseed_background.sh

real-paper-heterog-poisson-multiseed-dry-run:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json} --dry-run

real-paper-heterog-poisson-multiseed-process:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json}

real-paper-heterog-poisson-multiseed-process-partial:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json} --allow-partial

real-paper-heterog-poisson-multiseed-status:
	python experiments/real_matrix/status_multiseed.py --config $${CONFIG:-configs/tpds_multiseed_heterog_poisson.json}

real-paper-supplement-multiseed:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_second_model.json}

real-paper-supplement-multiseed-bg:
	CONFIG=$${CONFIG:-configs/tpds_multiseed_second_model.json} bash experiments/real_matrix/scripts/run_tpds_multiseed_background.sh

real-paper-supplement-multiseed-dry-run:
	python experiments/real_matrix/run_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_second_model.json} --dry-run

real-paper-supplement-multiseed-process:
	python experiments/real_matrix/process_multiseed_validation.py --config $${CONFIG:-configs/tpds_multiseed_second_model.json}

real-paper-supplement-multiseed-status:
	python experiments/real_matrix/status_multiseed.py --config $${CONFIG:-configs/tpds_multiseed_second_model.json}
