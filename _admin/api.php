<?php
/**
 * The admin page's write path: queue a run of update-benchmarks.yml.
 *
 * Deliberately small. Reads need no server at all -- llm.json comes from the
 * published site and _pending/pending.json from raw.githubusercontent, both of
 * which send Access-Control-Allow-Origin: * -- so the only thing that has to
 * live here is the credential, and the only thing it may do is dispatch this
 * one workflow and read its runs.
 *
 * The token is a fine-grained PAT scoped to the one repository with Actions:
 * read and write and nothing else. That matters more than it looks: a
 * contents-write token would bypass the branch ruleset outright, because the
 * repository owner is on its bypass list, which would turn this endpoint into
 * arbitrary-write-to-main. With Actions only, the worst an attacker who gets
 * past the host's auth can do is queue a workflow whose every record answer.py
 * validates.
 *
 * Authentication is the web server's job -- see .htaccess. Nothing here tries
 * to do it again.
 */

declare(strict_types=1);

const WORKFLOW = 'update-benchmarks.yml';
const REF      = 'main';           // never taken from the client: see the workflow
const MAX_RECORDS = 25;
const MAX_BYTES   = 60000;         // workflow_dispatch caps an input near 64 KB
const GUARD_HEADER = 'HTTP_X_AI_BENCH_ADMIN';

function fail(int $status, string $message): never
{
    http_response_code($status);
    header('Content-Type: application/json');
    echo json_encode(['error' => $message], JSON_UNESCAPED_SLASHES), "\n";
    exit;
}

function ok(array $body): never
{
    header('Content-Type: application/json');
    echo json_encode($body, JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT), "\n";
    exit;
}

/** Config lives outside the docroot so a misconfigured server cannot serve it. */
function config(): array
{
    $path = getenv('AI_BENCH_ADMIN_CONFIG')
        ?: dirname(__DIR__, 2) . '/ai-bench-admin-config.php';
    if (!is_readable($path)) {
        fail(500, 'Server is not configured: see _admin/README.md.');
    }
    $config = require $path;
    if (empty($config['token']) || empty($config['repo'])) {
        fail(500, 'Server config is missing a token or a repo.');
    }
    return $config;
}

/**
 * One GitHub API call. Returns [status, decoded body].
 *
 * The token goes in a header and never into a message: a failure is reported
 * with GitHub's own `message` field, which is what actually helps -- a
 * fine-grained PAT expires within a year and fails opaquely otherwise.
 */
function github(array $config, string $method, string $path, ?array $body = null): array
{
    $handle = curl_init('https://api.github.com/repos/' . $config['repo'] . $path);
    $headers = [
        'Accept: application/vnd.github+json',
        'Authorization: Bearer ' . $config['token'],
        'X-GitHub-Api-Version: 2022-11-28',
        'User-Agent: ai-bench-admin',
    ];
    if ($body !== null) {
        $headers[] = 'Content-Type: application/json';
        curl_setopt($handle, CURLOPT_POSTFIELDS, json_encode($body));
    }
    curl_setopt_array($handle, [
        CURLOPT_CUSTOMREQUEST  => $method,
        CURLOPT_HTTPHEADER     => $headers,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 20,
    ]);
    $raw = curl_exec($handle);
    if ($raw === false) {
        $error = curl_error($handle);
        curl_close($handle);
        fail(502, 'Could not reach GitHub: ' . $error);
    }
    $status = (int) curl_getinfo($handle, CURLINFO_HTTP_CODE);
    curl_close($handle);
    return [$status, json_decode((string) $raw, true) ?: []];
}

function recent_runs(array $config): array
{
    [$status, $body] = github(
        $config,
        'GET',
        '/actions/workflows/' . WORKFLOW . '/runs?per_page=8'
    );
    if ($status >= 400) {
        fail($status, $body['message'] ?? 'GitHub rejected the request.');
    }
    return array_map(static fn(array $run): array => [
        'id'         => $run['id'],
        'name'       => $run['display_title'] ?? $run['name'],
        'status'     => $run['status'],
        'conclusion' => $run['conclusion'],
        'event'      => $run['event'],
        'created_at' => $run['created_at'],
        'url'        => $run['html_url'],
    ], $body['workflow_runs'] ?? []);
}

// --------------------------------------------------------------------------

$config = config();
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';

if ($method === 'GET') {
    ok(['runs' => recent_runs($config)]);
}

if ($method !== 'POST') {
    fail(405, 'Only GET and POST.');
}

// A browser attaches the host's auth credentials to any cross-site request it
// makes, so a hostile page could otherwise post here on your behalf. Requiring
// a JSON content type and a custom header forces a preflight, which a simple
// cross-site form cannot produce.
if (!str_starts_with($_SERVER['CONTENT_TYPE'] ?? '', 'application/json')) {
    fail(415, 'Send application/json.');
}
if (empty($_SERVER[GUARD_HEADER])) {
    fail(400, 'Missing the X-AI-Bench-Admin header.');
}

$raw = file_get_contents('php://input') ?: '';
if (strlen($raw) > MAX_BYTES) {
    fail(413, 'That batch is too large; send it in smaller pieces.');
}
$request = json_decode($raw, true);
if (!is_array($request) || !isset($request['answers']) || !is_array($request['answers'])) {
    fail(400, 'Expected {"answers": [...]}.');
}
$answers = array_values($request['answers']);
if ($answers === []) {
    fail(400, 'No answers to send.');
}
// Enforced here as well as in the page, because the page is not the boundary --
// and again in answer.py, which is the one that actually counts.
if (count($answers) > MAX_RECORDS) {
    fail(400, 'At most ' . MAX_RECORDS . ' answers per batch.');
}

// The workflow's concurrency group holds one run plus one queued; a third
// arrival cancels the queued one, silently, before it does any work. Refusing
// while one is already waiting is what stops a second batch evicting the first.
foreach (recent_runs($config) as $run) {
    if ($run['status'] === 'queued') {
        fail(409, 'A run is already queued; wait for it to start, then send this batch.');
    }
}

$payload = json_encode($answers, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
[$status, $body] = github($config, 'POST', '/actions/workflows/' . WORKFLOW . '/dispatches', [
    'ref'    => REF,
    'inputs' => [
        'answers'      => $payload,
        'skip_refresh' => !empty($request['skip_refresh']) ? 'true' : 'false',
    ],
]);
if ($status >= 400) {
    fail($status, $body['message'] ?? 'GitHub rejected the dispatch.');
}

ok(['dispatched' => count($answers)]);
