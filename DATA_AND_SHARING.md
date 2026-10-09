# Data and Sharing Boundaries

This is a code-focused public review repository. It excludes material that is unnecessary for understanding the research workflows or cannot be redistributed as part of this repository.

Excluded material includes:

- API keys, access tokens, `.env` files, and historical Git commits containing credentials;
- downloaded satellite imagery and generated image tiles;
- coordinate datasets, KML and KMZ files, and extracted geographic records;
- model outputs, run directories, evaluation results, and generated reports;
- training datasets and generated fine-tuning artifacts;
- local virtual environments, caches, bytecode, and operating-system metadata;
- hard-coded personal filesystem paths;
- private model, file, evaluation, and job identifiers;
- one-off CSV/JSON conversion, result-flattening, batch-rerun, and repair scripts; and
- the manuscript.

A reviewer who wishes to execute the workflows must separately provide authorized imagery access, input data, provider credentials, models available to the reviewer's account, and output locations outside the repository.

The source repository historically contained credentials. This public repository was created with fresh Git history and contains no known credential values. Credentials previously used in the source project should be treated as exposed independently of this clean copy.

Public visibility does not grant an open-source license or permission to redistribute excluded research material.
