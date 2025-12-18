Notes for AI. 

General setup.
- When setting up a new experiment directory, always create a `.gitignore` file that ignores the `training_data/` folder. This prevents large generated datasets from being committed to git.

When writing unit tests
- Default to writing tests as pure functions with extremely descriptive names.
- Use conftest.py as needed to define fixtures that will be reused across multiple tests.