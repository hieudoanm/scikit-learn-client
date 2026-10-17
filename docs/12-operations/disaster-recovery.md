# Disaster Recovery

Local-only. Disaster = disk loss or `docker compose down -v`.

Recovery:

1. Re-clone / checkout the repo.
2. Restore `backup.sql` and `skl_data.tgz` if they exist.
3. `docker compose up --build`.
4. Confirm `/readyz` and a historical job download.

If no backup exists, retrain from CSVs. RTO: time to rebuild images (minutes). Document that this is not a HA design.
