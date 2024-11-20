"""Add language column to CommentHistory model

Revision ID: 2f7d316e7fe3
Revises: 
Create Date: 2024-11-20 19:04:57.319279

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2f7d316e7fe3'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('comment_history', schema=None) as batch_op:
        batch_op.add_column(sa.Column('language', sa.String(length=10), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('comment_history', schema=None) as batch_op:
        batch_op.drop_column('language')

    # ### end Alembic commands ###