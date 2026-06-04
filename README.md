# MilkingTime

A Streamlit demo app for dairy farm shift timekeeping with:

- Worker self-enrollment using 3 face captures
- Face-vector based check-in and check-out
- Geo-fence status tracking with a configurable grace period
- Automatic checkout after the grace period when a worker is outside the fence
- Manager dashboard protected by an admin PIN
- Weekly hours, weekend overtime, hourly rates, and salary estimates in South African Rand
- Local SQLite storage now, with a Supabase-ready data shape for later

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

```bash
export MANAGER_PIN="your-pin"
```

## Supabase

The first version runs locally with SQLite at `.milkingtime/milkingtime.db`.
When your Supabase project is ready, set these environment variables:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
```

The app currently keeps local storage as the active backend so the demo remains self-contained. The table design mirrors the records that should be created in Supabase:

- `workers`
- `face_embeddings`
- `shifts`
- `settings`

## Geo-fence

Until the exact farm geo-fence is known, managers can configure a temporary center latitude, longitude, radius, and grace period in the manager screen. Workers enter their current latitude and longitude in the demo flow.

For a production mobile app, browser/device GPS should be captured automatically and checked continuously or on a regular heartbeat.

