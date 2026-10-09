# Data and Sharing Boundaries

This is a code-only public review repository. It excludes material that is unnecessary for understanding the pipeline or cannot be redistributed as part of this repository.

Excluded material includes:

- API keys, access tokens, `.env` files, and historical Git commits containing credentials;
- downloaded satellite imagery and generated image tiles;
- coordinate datasets, KML and KMZ files, and extracted geographic records;
- model outputs, run directories, evaluation results, and generated reports;
- training data and fine-tuning artifacts;
- local virtual environments, caches, bytecode, and operating-system metadata;
- hard-coded personal filesystem paths;
- private model, file, evaluation, and job identifiers; and
- the manuscript.

A reviewer who wishes to execute the pipeline must separately provide authorized imagery access, input coordinates, provider credentials, an available vision-capable model, and an output location outside the repository.

The source repository historically contained credentials. This public repository was created with fresh Git history and contains no known credential values. Credentials previously used in the source project should be treated as exposed independently of this clean copy.

Public visibility does not grant an open-source license or permission to redistribute excluded research material.
