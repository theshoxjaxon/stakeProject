"""add_sport_key_to_matches

Revision ID: b5f9e2a4c7d1
Revises: c09822358c19
Create Date: 2026-04-22 00:00:00.000000

Adds sport_key (nullable) to the matches table so the Credit Protector
can query the latest odds timestamp per sport without a full-table scan.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5f9e2a4c7d1'
down_revision: Union[str, Sequence[str], None] = 'c09822358c19'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add sport_key column and index to matches."""
    op.add_column(
        'matches',
        sa.Column('sport_key', sa.String(50), nullable=True),
    )
    op.create_index('ix_matches_sport_key', 'matches', ['sport_key'])


def downgrade() -> None:
    """Remove sport_key column and index from matches."""
    op.drop_index('ix_matches_sport_key', table_name='matches')
    op.drop_column('matches', 'sport_key')
