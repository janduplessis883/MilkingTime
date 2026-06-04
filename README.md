# MilkingTime

A Streamlit demo app for dairy farm shift timekeeping with:

- Worker self-enrollment using 3 face captures
- Face-vector based check-in and check-out
- Geo-fence status tracking with a configurable grace period
- Automatic checkout after the grace period when a worker is outside the fence
- Manager dashboard protected by an admin PIN
- Weekly hours, weekend overtime, hourly rates, and salary estimates in South African Rand
- Supabase storage when configured, with local SQLite fallback for demos

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo Admin PIN

The default manager PIN is:

```text
1234
```

Override it by setting:

```toml
[app]
MANAGER_PIN = "your-pin"
```

## Supabase

The app follows the Streamlit Supabase setup pattern and reads credentials from
`.streamlit/secrets.toml`. That file is ignored by git.

```toml
[connections.supabase]
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-api-key"
```

Before using Supabase, open the SQL editor in Supabase and run
`supabase_schema.sql`. It creates:

- `workers`
- `face_embeddings`
- `shifts`
- `settings`

If Streamlit secrets are missing, the app falls back to local SQLite storage at
`.milkingtime/milkingtime.db`.

## Geo-fence

Until the exact farm geo-fence is known, managers can configure a temporary center latitude, longitude, radius, and grace period in the manager screen. Workers enter their current latitude and longitude in the demo flow.

For a production mobile app, browser/device GPS should be captured automatically and checked continuously or on a regular heartbeat.
