# DataJoint Test Environment

A local DataJoint MySQL server for development and testing. Lets you exercise the full acquisition → DataJoint upload path without touching the production database.

## When to use it

- Developing changes to DataJoint table definitions or `populate()` logic
- Verifying that a session uploads cleanly before pointing at production DJ
- Running CI-style smoke tests of the upload path

If you only need to run analysis code that reads from production DJ, you don't need this — just configure `datajoint_config.json` with production credentials.

## Architecture

```
   ┌──────────────────────────────┐         ┌────────────────────────────┐
   │      mocap-test-dj           │         │      datajoint-test        │
   │      (isr/mct_acquisition)   │         │      (mysql:8.0)           │
   │                              │         │                            │
   │  /root/.datajoint_config.json│ ───────▶│  127.0.0.1:3307            │
   │   ↑ mounted read-only from   │  MySQL  │  root / djtest             │
   │   docker/datajoint_config    │         │                            │
   │   .test.json                 │         │  Volume: datajoint_test_db │
   │                              │         │  (persists across restarts)│
   │  /datajoint_external         │         └────────────────────────────┘
   │   ↑ mounted from
   │   ${TEST_DJ_EXTERNAL:-/tmp/datajoint_external_test}
   └──────────────────────────────┘
```

Two containers cooperate:

- **`datajoint-test`** — MySQL 8.0 bound to `127.0.0.1:3307`. Data persists in the `datajoint_test_db` named volume across restarts. Healthcheck (`mysqladmin ping`) gates dependent services.
- **`mocap-test-dj`** — Same image as `mocap`, but with `docker/datajoint_config.test.json` mounted over the in-image `datajoint_config.json`. Points at `127.0.0.1:3307`, root user, password `djtest`. External storage redirects to `${TEST_DJ_EXTERNAL:-/tmp/datajoint_external_test}`.

## First-time setup

```bash
make init-dj-test
```

Creates the external-storage directory and writes a sentinel file (`.multi_cam_mount_check`) that the DataJoint mount-check logic uses to verify the external store is mounted. Override the path with `TEST_DJ_EXTERNAL`:

```bash
make init-dj-test TEST_DJ_EXTERNAL=/path/to/external
```

You only need to run this once per workstation.

## Daily workflow

```bash
make run-mocap-test-dj
```

This starts `datajoint-test`, waits for its health check to pass, then launches `mocap-test-dj` in the foreground against it. Ctrl-C stops the acquisition container; **the MySQL container keeps running** so subsequent launches start fast.

Cannot run simultaneously with `make run` or `make run-mocap-test` (all three bind host ports 8000 and 3000).

## Cleanup

| Goal | Command | Effect |
|---|---|---|
| Stop MySQL, keep data | `docker compose stop datajoint-test` | Container stops; `datajoint_test_db` volume preserved; resumes on next `make run-mocap-test-dj` |
| Drop and recreate (clean slate) | `make reset-dj-test` | Stops container, removes it, deletes the `datajoint_test_db` volume, restarts fresh |

> **Note:** A `make stop-dj-test` shortcut is tracked as issue #21 — it has not landed yet. Use `docker compose stop datajoint-test` directly in the meantime.

`make reset-dj-test` is the right call when:
- Schema changed and you want a fresh DB
- A failed insert left data in an inconsistent state
- You want to confirm a setup script works from zero

## Configuration

### DataJoint config

`docker/datajoint_config.test.json` is mounted read-only over the container's `~/.datajoint_config.json`. Defaults:

```json
{
    "database.host": "127.0.0.1",
    "database.port": 3307,
    "database.user": "root",
    "database.password": "djtest",
    "enable_python_native_blobs": true,
    "stores": { "localattach": { "protocol": "file", "location": "/datajoint_external" } }
}
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DJ_TEST_PASS` | `djtest` | MySQL root password. **If you change this, also update `database.password` in `datajoint_config.test.json`.** |
| `TEST_DJ_EXTERNAL` | `/tmp/datajoint_external_test` | Host path for DataJoint external blob storage |
| `TEST_DATA_VOLUME` | `/data-test` | Host path for recording data |

## Troubleshooting

**`make run-mocap-test-dj` hangs at "waiting for datajoint-test to be healthy"**

The MySQL container's healthcheck has a 30-second start period, then polls every 10 s with up to 5 retries. If it never becomes healthy, check the logs:

```bash
docker compose logs datajoint-test
```

Common causes: another process is bound to `127.0.0.1:3307`, or a previous `datajoint_test_db` volume was created with a different `MYSQL_ROOT_PASSWORD` (MySQL persists the root password on first init — see "Drop and recreate" above).

**`port 3307 already in use`**

Another process is bound there. Identify it:

```bash
sudo lsof -i :3307
```

Either stop that process or change the host port in `docker-compose.yml` (and update `database.port` in `datajoint_config.test.json` to match).

**`Mount point not detected`**

The external-storage sentinel is missing. Re-run `make init-dj-test`.

**Schema looks stale after pulling new code**

Run `make reset-dj-test` to drop the database and let the new schema create cleanly on next launch.

## See also

- [Container Reference](container_reference.md) — `datajoint-test` and `mocap-test-dj` service details
- [Backend Architecture](../development/backend_architecture.md) — how the dual-DB pattern works
