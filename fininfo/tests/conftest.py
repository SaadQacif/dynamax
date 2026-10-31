"""Numerical configuration shared by the financial-information tests."""

import jax


jax.config.update("jax_enable_x64", True)
