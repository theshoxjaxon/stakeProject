"""add_multitenant_schema

Revision ID: 061f399509fd
Revises: b5f9e2a4c7d1
Create Date: 2026-09-14 12:52:02.303847

B2B multi-tenant schema: organizations (tenant), users (belong to one org,
`role` is per-user), api_keys and usage_events (org-scoped), refresh_tokens
(user-scoped). Only hashes of API keys / refresh tokens are ever stored.

`role` (per-user permission) and `plan` (org billing tier) are deliberately
separate columns on separate tables, enforced via CHECK constraints rather
than a native ENUM type — avoids the enum-type-survives-downgrade trap
(Postgres ENUM types aren't dropped automatically when the owning
column/table is, which breaks a downgrade -> upgrade round trip).

Also denormalizes `kickoff` / `league` onto `predictions` (copied from
`matches.date` / `matches.sport_key`) so the "today's predictions" query
can hit one composite index instead of joining matches.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '061f399509fd'
down_revision: Union[str, Sequence[str], None] = 'b5f9e2a4c7d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the multi-tenant tables and the predictions (kickoff, league) index."""
    # citext (case-insensitive text) backs users.email; must exist before create_table.
    op.execute('CREATE EXTENSION IF NOT EXISTS citext')

    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('plan', sa.String(length=20), server_default='free', nullable=False),
        sa.Column('plan_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.CheckConstraint("plan IN ('free', 'pro', 'premium')", name='ck_organizations_plan'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', postgresql.CITEXT(), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=20), server_default='member', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.CheckConstraint("role IN ('owner', 'admin', 'member')", name='ck_users_role'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    op.create_index('ix_users_org_id', 'users', ['org_id'])

    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('prefix', sa.String(length=12), nullable=False),
        sa.Column(
            'scopes', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False
        ),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash'),
    )
    op.create_index('ix_api_keys_org_id', 'api_keys', ['org_id'])

    op.create_table(
        'refresh_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('replaced_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['replaced_by'], ['refresh_tokens.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash'),
    )
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])

    op.create_table(
        'usage_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column(
            'ts', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False
        ),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_usage_events_org_ts', 'usage_events', ['org_id', 'ts'])

    # predictions: denormalized kickoff/league for the "today" query.
    op.add_column('predictions', sa.Column('kickoff', sa.DateTime(), nullable=True))
    op.add_column('predictions', sa.Column('league', sa.String(length=50), nullable=True))
    op.execute(
        """
        UPDATE predictions
        SET kickoff = matches.date,
            league = matches.sport_key
        FROM matches
        WHERE predictions.match_id = matches.id
        """
    )
    op.create_index('ix_predictions_kickoff_league', 'predictions', ['kickoff', 'league'])


def downgrade() -> None:
    """Drop the predictions index/columns and the multi-tenant tables, in dependency order."""
    op.drop_index('ix_predictions_kickoff_league', table_name='predictions')
    op.drop_column('predictions', 'league')
    op.drop_column('predictions', 'kickoff')

    op.drop_index('ix_usage_events_org_ts', table_name='usage_events')
    op.drop_table('usage_events')

    op.drop_index('ix_refresh_tokens_user_id', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')

    op.drop_index('ix_api_keys_org_id', table_name='api_keys')
    op.drop_table('api_keys')

    op.drop_index('ix_users_org_id', table_name='users')
    op.drop_table('users')

    op.drop_table('organizations')

    op.execute('DROP EXTENSION IF EXISTS citext')
