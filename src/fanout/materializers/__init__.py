from fanout.materializers.base import BaseMaterializer, register_materializer, get_materializer, list_materializers  # noqa: F401

# Import built-in materializers to trigger registration
import fanout.materializers.file  # noqa: F401
import fanout.materializers.stdin  # noqa: F401
import fanout.materializers.worktree  # noqa: F401
