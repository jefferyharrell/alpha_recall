"""Template loading service for Jinja templates."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

from ..logging import get_logger

logger = get_logger(__name__)


class TemplateLoader:
    """Service for loading and rendering Jinja templates."""

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize the template loader.

        Args:
            template_dir: Directory containing template files.
                         Defaults to prompts/ directory in project root.
        """
        if template_dir is None:
            # Default to prompts/ directory in project root
            project_root = Path(__file__).parent.parent.parent.parent
            template_dir = project_root / "prompts"

        self.template_dir = Path(template_dir)

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        logger.debug(f"Template loader initialized with directory: {self.template_dir}")

    def load_template(self, template_name: str) -> Template:
        """Load a template by name.

        Args:
            template_name: Name of the template file

        Returns:
            Jinja2 Template object

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        try:
            template = self.env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            raise

    def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Load and render a template with given context.

        Args:
            template_name: Name of the template file
            context: Dictionary of variables to pass to template

        Returns:
            Rendered template as string
        """
        template = self.load_template(template_name)

        try:
            rendered = template.render(**context)
            logger.debug(
                f"Rendered template {template_name} with context keys: {list(context.keys())}"
            )
            return rendered
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise


# Global template loader instance
template_loader = TemplateLoader()
