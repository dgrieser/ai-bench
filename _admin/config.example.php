<?php
/**
 * Copy this OUTSIDE the docroot -- api.php looks for it at
 * ../../ai-bench-admin-config.php relative to itself, or wherever
 * $AI_BENCH_ADMIN_CONFIG points -- and fill in the token.
 *
 * Never commit the real one. The example is here so the shape is documented;
 * api.php refuses to start without a token rather than falling back to
 * anything.
 *
 * The token is a FINE-GRAINED personal access token:
 *
 *   Repository access : only dgrieser/ai-bench
 *   Permissions       : Actions -> Read and write
 *                       (nothing else -- see the note in api.php about why
 *                        Contents: write would be a much worse trade)
 *
 * With Actions alone the token can queue this one workflow and read its runs.
 * It cannot write a file, so every change still goes through answer.py's
 * validation on the runner.
 *
 * Fine-grained tokens expire within a year. When dispatches start failing, that
 * is usually why; api.php passes GitHub's own message through so you can see it.
 */

return [
    'token' => 'github_pat_...',
    'repo'  => 'dgrieser/ai-bench',
];
