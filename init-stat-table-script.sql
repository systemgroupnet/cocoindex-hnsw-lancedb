-- init-stat-table-script.sql
--
-- Creates the MySQL tables that cocoindex-code pushes index statistics into,
-- for building metrics dashboards in Apache DevLake (Grafana over MySQL).
--
-- The daemon writes one snapshot after every index pass when a MySQL target is
-- configured (see the COCOINDEX_CODE_METRICS_* environment variables). Each
-- snapshot is one row in ccc_repo_stats plus one row per language in
-- ccc_language_stats, all sharing the same `collected_at` timestamp and
-- `snapshot_id`, so panels can chart totals over time and break a point in time
-- down by language.
--
-- Usage:
--   mysql -h <host> -P <port> -u <user> -p <database> < init-stat-table-script.sql
--
-- Safe to run more than once (CREATE TABLE IF NOT EXISTS).

-- Repo-level totals: one row per index pass (time-series keyed by repo).
CREATE TABLE IF NOT EXISTS ccc_repo_stats (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    snapshot_id  CHAR(32)     NOT NULL COMMENT 'Correlates this repo row with its language rows',
    repo         VARCHAR(512) NOT NULL COMMENT 'Repo identifier (host path or COCOINDEX_CODE_METRICS_REPO)',
    collected_at DATETIME(6)  NOT NULL COMMENT 'UTC timestamp of the snapshot',
    total_chunks BIGINT       NOT NULL,
    total_files  BIGINT       NOT NULL,
    total_loc    BIGINT       NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_repo_snapshot (snapshot_id),
    KEY idx_repo_time (repo, collected_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Per-language breakdown: one row per language per index pass.
CREATE TABLE IF NOT EXISTS ccc_language_stats (
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    snapshot_id  CHAR(32)     NOT NULL COMMENT 'Matches ccc_repo_stats.snapshot_id',
    repo         VARCHAR(512) NOT NULL,
    collected_at DATETIME(6)  NOT NULL COMMENT 'UTC timestamp of the snapshot (same as the repo row)',
    language     VARCHAR(128) NOT NULL,
    chunks       BIGINT       NOT NULL,
    loc          BIGINT       NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uq_lang_snapshot (snapshot_id, language),
    KEY idx_repo_lang_time (repo, language, collected_at),
    KEY idx_snapshot (snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
