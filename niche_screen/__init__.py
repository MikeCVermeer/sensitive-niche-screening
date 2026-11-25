"""Synthetic training demo and frozen research model inference."""
from .inference import load_model, score_pages, score_site

__all__ = ['load_model', 'score_pages', 'score_site']
