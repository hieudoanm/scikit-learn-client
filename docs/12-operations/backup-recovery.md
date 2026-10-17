# Backup and Recovery

## PostgreSQL (local)

```bash
docker compose exec postgres pg_dump -U skl skl > backup.sql
docker compose exec -T postgres psql -U skl skl < backup.sql
```

## Artefact volume

```bash
docker run --rm -v skl_data:/data -v "$(pwd)":/backup alpine tar czf /backup/skl_data.tgz /data
```

Restore by extracting into the named volume. Keep dump and tarball together — jobs without files are useless.

There is no managed PITR. RPO is "last manual backup".
