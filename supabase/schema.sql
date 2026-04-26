create table if not exists tracks (
  id bigserial primary key,
  track_key text not null unique,
  provider text not null,
  provider_track_id text null,
  name text not null,
  country_code text null,
  timezone text null default 'Europe/London',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_tracks_provider on tracks(provider);

create table if not exists owners (
  id bigserial primary key,
  slug text not null unique,
  name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists trainers (
  id bigserial primary key,
  slug text not null unique,
  name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists dogs (
  id bigserial primary key,
  dog_key text not null unique,
  provider text not null,
  provider_dog_id text null,
  name text not null,
  sex text null,
  date_of_birth date null,
  sire_name text null,
  dam_name text null,
  owner_id bigint null references owners(id),
  trainer_id bigint null references trainers(id),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_dogs_owner_id on dogs(owner_id);
create index if not exists idx_dogs_trainer_id on dogs(trainer_id);

create table if not exists races (
  id bigserial primary key,
  race_key text not null unique,
  provider text not null,
  provider_race_id text null,
  meeting_id text null,
  track_id bigint not null references tracks(id),
  scheduled_start timestamptz not null,
  off_time timestamptz null,
  race_number integer null,
  race_name text null,
  distance_m integer null,
  grade text null,
  going text null,
  purse numeric(12,2) null,
  status text not null default 'scheduled',
  is_completed boolean not null default false,
  metadata_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_races_track_time on races(track_id, scheduled_start);
create index if not exists idx_races_completed on races(is_completed, scheduled_start);

create table if not exists race_entries (
  id bigserial primary key,
  entry_key text not null unique,
  race_id bigint not null references races(id) on delete cascade,
  dog_id bigint null references dogs(id),
  trap_number integer not null,
  weight_kg numeric(8,3) null,
  finish_position integer null,
  official_time_s numeric(8,3) null,
  sectional_s numeric(8,3) null,
  beaten_distance text null,
  sp_text text null,
  sp_decimal numeric(10,4) null,
  comment text null,
  vacant boolean not null default false,
  scratched boolean not null default false,
  metadata_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (race_id, trap_number)
);

create index if not exists idx_race_entries_race_id on race_entries(race_id);
create index if not exists idx_race_entries_dog_id on race_entries(dog_id);
create index if not exists idx_race_entries_finish on race_entries(finish_position);

create table if not exists raw_fetches (
  id bigserial primary key,
  source text not null,
  source_url text not null,
  fetched_at timestamptz not null default now(),
  http_status integer null,
  content_sha256 text not null,
  payload_text text not null
);

create index if not exists idx_raw_fetches_source_time on raw_fetches(source, fetched_at desc);

create table if not exists ingestion_runs (
  id bigserial primary key,
  source text not null,
  started_at timestamptz not null default now(),
  finished_at timestamptz null,
  status text not null default 'running',
  rows_processed integer not null default 0,
  races_touched integer not null default 0,
  notes_json jsonb not null default '{}'::jsonb,
  error_text text null
);

create index if not exists idx_ingestion_runs_source on ingestion_runs(source, started_at desc);

create table if not exists training_runs (
  id bigserial primary key,
  run_key text not null unique,
  started_at timestamptz not null default now(),
  finished_at timestamptz null,
  status text not null default 'running',
  config_json jsonb not null default '{}'::jsonb,
  metrics_json jsonb not null default '{}'::jsonb,
  artifact_path text null,
  report_path text null,
  train_race_count integer not null default 0,
  validation_race_count integer not null default 0,
  error_text text null
);

create index if not exists idx_training_runs_status on training_runs(status, started_at desc);

create table if not exists prediction_runs (
  id bigserial primary key,
  run_key text not null unique,
  race_id bigint not null references races(id) on delete cascade,
  training_run_id bigint null references training_runs(id),
  created_at timestamptz not null default now(),
  predicted_order_json jsonb not null default '[]'::jsonb,
  confidence numeric(10,6) null,
  metadata_json jsonb not null default '{}'::jsonb
);

create index if not exists idx_prediction_runs_race on prediction_runs(race_id, created_at desc);

create table if not exists prediction_entries (
  id bigserial primary key,
  prediction_run_id bigint not null references prediction_runs(id) on delete cascade,
  race_entry_id bigint null references race_entries(id),
  dog_id bigint null references dogs(id),
  predicted_rank integer not null,
  score numeric(12,6) not null,
  win_probability numeric(12,6) null,
  created_at timestamptz not null default now()
);

create index if not exists idx_prediction_entries_run on prediction_entries(prediction_run_id, predicted_rank);
