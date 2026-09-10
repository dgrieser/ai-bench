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

// Bumped whenever the shape the page depends on changes, so a half-updated
// deployment says so instead of rendering an empty, unexplained page. The page
// refuses to run below NEEDS_API; anything added since is announced in the GET
// payload instead, so a newer page against an older endpoint loses the new
// thing rather than the whole queue.
const API_VERSION = 4;

const WORKFLOW = 'update-benchmarks.yml';
// The workflow step that applies a dispatched batch, by name. The run's own
// conclusion cannot stand in for it: see answer_step() below.
const ANSWER_STEP = 'Apply the answers';
// The legacy /api/v2/data/llms/models route answers 410 Gone from 2026-11-04
// (https://artificialanalysis.ai/data-api/migrate-v2-data). Only the slugs are
// read here and both tiers carry those, so a key without a Pro subscription
// falls back to the free route rather than costing the page its suggestions.
const AA_MODELS_URL      = 'https://artificialanalysis.ai/api/v2/language/models';
const AA_MODELS_FREE_URL = 'https://artificialanalysis.ai/api/v2/language/models/free';
// The list endpoint pages at 200 records; the cap guards against a response
// that keeps claiming another page, not against AA's ~600 models.
const AA_MAX_PAGES = 25;
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

/**
 * Where the config may live. It has to be outside the docroot, so that a server
 * that stops running PHP -- a bad .htaccess, a module falling over -- serves a
 * 404 rather than the token in plain text.
 *
 * Hosts disagree about the shape above the docroot, so rather than guess once
 * and fail obscurely, this tries the sensible candidates in order. Set
 * AI_BENCH_ADMIN_CONFIG to skip the guessing; SetEnv in .htaccess lands in
 * $_SERVER under mod_php and in the environment under FPM, so both are read.
 */
function config_candidates(): array
{
    $explicit = $_SERVER['AI_BENCH_ADMIN_CONFIG'] ?? getenv('AI_BENCH_ADMIN_CONFIG');
    if (is_string($explicit) && $explicit !== '') {
        return [$explicit];
    }
    $name = '/ai-bench-admin-config.php';
    $candidates = [dirname(__DIR__, 2) . $name, dirname(__DIR__, 3) . $name];
    if (!empty($_SERVER['DOCUMENT_ROOT'])) {
        // The likeliest of the lot: one level above the docroot itself, whatever
        // depth this directory happens to sit at.
        array_unshift($candidates, dirname($_SERVER['DOCUMENT_ROOT']) . $name);
    }
    return array_values(array_unique($candidates));
}

function config(): array
{
    $tried = config_candidates();
    foreach ($tried as $path) {
        if (!is_readable($path)) {
            continue;
        }
        $config = require $path;
        if (!is_array($config) || empty($config['token']) || empty($config['repo'])) {
            error_log("ai-bench admin: $path has no token or repo");
            fail(500, 'Server config is incomplete; see _admin/README.md.');
        }
        return $config;
    }
    // The paths go to the log, not the response: if the .htaccess is what is
    // broken, this reply is not behind any authentication.
    error_log('ai-bench admin: no config found; tried ' . implode(', ', $tried));
    fail(500, 'Server is not configured. See _admin/README.md; the paths tried are in the error log.');
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

/**
 * Where the page should read its data, derived rather than configured.
 *
 * Both files come from raw.githubusercontent, not from the published site. The
 * site would work for llm.json, but only under its custom domain: the
 * dgrieser.github.io form 301s to openbench.david-grieser.de, and a
 * cross-origin redirect carries no CORS headers, so a derived github.io URL
 * fails in the browser. llm.html has the same note at its DATA_URLS.
 *
 * Reading both from raw means the repo name is the only thing configured, and
 * the page sees main exactly as the last run left it rather than whatever the
 * Pages build has caught up with.
 */
function data_base(array $config): string
{
    return 'https://raw.githubusercontent.com/' . $config['repo'] . '/' . REF;
}

/**
 * Every Artificial Analysis model slug, sorted. Names only -- nothing else on
 * those records is any of the page's business.
 *
 * Why this is proxied rather than fetched in the browser: the AA API is
 * authenticated with a header, so a cross-origin request preflights, and AA
 * answers no CORS headers -- the fetch fails before it is sent. Routing it
 * through here also keeps the key on the server, where the GitHub token already
 * lives, instead of shipping it to every browser that opens the page.
 *
 * The key is optional. Without it the page simply does not offer the slug list,
 * which is why this is announced in the GET payload rather than required.
 */
function aa_slugs(array $config): array
{
    $key = $config['aa_api_key'] ?? '';
    if (!is_string($key) || $key === '') {
        fail(501, 'No Artificial Analysis API key in the config; see _admin/README.md.');
    }

    [$slugs, $status] = aa_slug_pages(AA_MODELS_URL, $key);
    if ($status === 403) {
        // A key without a Pro subscription. The free route lists the same
        // models with fewer fields on each, and fields are not what this reads.
        [$slugs, $status] = aa_slug_pages(AA_MODELS_FREE_URL, $key);
    }
    if ($status >= 400) {
        // An expired or wrong key is the usual reason, and it fails opaquely
        // otherwise -- the same trap the GitHub token has.
        fail($status, 'Artificial Analysis rejected the request (status ' . $status . ').');
    }

    $slugs = array_values(array_unique($slugs));
    sort($slugs);
    return $slugs;
}

/**
 * The slugs on one endpoint, as [slugs, status]. The V2 list endpoints page
 * their answers, so this walks them until one says there is no more.
 */
function aa_slug_pages(string $url, string $key): array
{
    $slugs = [];
    for ($page = 1; $page <= AA_MAX_PAGES; $page++) {
        $handle = curl_init($url . '?page=' . $page);
        curl_setopt_array($handle, [
            CURLOPT_HTTPHEADER     => ['x-api-key: ' . $key, 'User-Agent: ai-bench-admin'],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 20,
        ]);
        $raw = curl_exec($handle);
        if ($raw === false) {
            $error = curl_error($handle);
            curl_close($handle);
            fail(502, 'Could not reach Artificial Analysis: ' . $error);
        }
        $status = (int) curl_getinfo($handle, CURLINFO_HTTP_CODE);
        curl_close($handle);
        if ($status >= 400) {
            return [[], $status];
        }

        $body = json_decode((string) $raw, true);
        if (!is_array($body)) {
            return [[], 502];
        }
        foreach ($body['data'] ?? [] as $model) {
            if (is_array($model) && !empty($model['slug']) && is_string($model['slug'])) {
                $slugs[] = $model['slug'];
            }
        }
        if (empty($body['pagination']['has_more'])) {
            break;
        }
    }

    return [$slugs, 200];
}

/**
 * What the run's own answer step concluded, if that run has one.
 *
 * The run's conclusion cannot answer this. `Fail if a step of update-all
 * failed` runs last and reds a run whose answers were applied, committed and
 * pushed minutes earlier, while a batch answer.py refused fails a run that
 * wrote nothing at all. Only the step separates the two, and the page has to
 * know which happened: re-sending an answer that did land is refused as a
 * question nothing asked, and a batch is all or nothing, so one duplicate
 * costs the whole next sitting.
 *
 * `conclusion` is null when this run carried no answers -- a scheduled refresh,
 * or one whose input was empty -- which is not the same as a failure and is
 * reported as its own case.
 */
function answer_step(array $config, string $run): array
{
    [$status, $body] = github($config, 'GET', '/actions/runs/' . $run . '/jobs?per_page=30');
    if ($status >= 400) {
        fail($status, $body['message'] ?? 'GitHub rejected the request.');
    }
    foreach ($body['jobs'] ?? [] as $job) {
        foreach ($job['steps'] ?? [] as $step) {
            if (($step['name'] ?? '') === ANSWER_STEP) {
                return ['step' => ANSWER_STEP, 'conclusion' => $step['conclusion']];
            }
        }
    }
    return ['step' => ANSWER_STEP, 'conclusion' => null];
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
    // One extra read, asked for only when the page needs it: the AA payload is
    // far larger than everything else here put together, and most sittings
    // never open the rename control at all.
    if (($_GET['aa'] ?? '') === 'slugs') {
        ok(['api' => API_VERSION, 'slugs' => aa_slugs($config)]);
    }
    // Asked for once per dispatched batch, when its run finishes. Digits only:
    // the value is pasted into a GitHub path.
    $answered = (string) ($_GET['answered'] ?? '');
    if ($answered !== '') {
        if (preg_match('/^[0-9]{1,20}$/', $answered) !== 1) {
            fail(400, 'answered must be a run id.');
        }
        ok(['api' => API_VERSION] + answer_step($config, $answered));
    }
    ok([
        'api'  => API_VERSION,
        'repo' => $config['repo'],
        'ref'  => REF,
        'raw'  => data_base($config),
        // Whether the slug list above can be served at all, so the page can
        // offer the control or explain its absence instead of failing at it.
        'aa'   => !empty($config['aa_api_key']),
        // Whether ?answered=<run id> can be served. A page against an older
        // endpoint keeps every answered card locked rather than guessing from
        // the run's conclusion, which is the safe half of the choice.
        'steps' => true,
        'runs' => recent_runs($config),
    ]);
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
