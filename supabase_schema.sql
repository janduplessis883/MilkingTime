create table if not exists workers (
    id uuid primary key,
    full_name text not null,
    phone text,
    hourly_rate numeric not null,
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists face_embeddings (
    id uuid primary key,
    worker_id uuid not null references workers(id) on delete cascade,
    embedding jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists shifts (
    id uuid primary key,
    worker_id uuid not null references workers(id) on delete cascade,
    checked_in_at timestamptz not null,
    checked_out_at timestamptz,
    checkin_lat double precision,
    checkin_lon double precision,
    checkout_lat double precision,
    checkout_lon double precision,
    auto_checkout boolean not null default false,
    outside_since timestamptz,
    notes text
);

create table if not exists settings (
    key text primary key,
    value text not null
);

insert into settings (key, value)
values
    ('farm_lat', '-33.9249'),
    ('farm_lon', '18.4241'),
    ('farm_radius_m', '500'),
    ('grace_minutes', '10'),
    ('overtime_multiplier', '1.5')
on conflict (key) do nothing;

alter table workers disable row level security;
alter table face_embeddings disable row level security;
alter table shifts disable row level security;
alter table settings disable row level security;
