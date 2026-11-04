// Build search through Pagefind's CLI so its process lifecycle is explicit.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const { TOLA_HOOK_INPUT_DIR: site, TOLA_HOOK_OUTPUT_DIR: out } = process.env;
if (!site || !out) {
  console.error("Run through Tola's search hook: npm run build.");
  process.exit(1);
}
const cli = fileURLToPath(new URL("../node_modules/pagefind/lib/runner/bin.cjs", import.meta.url));
if (!existsSync(cli)) {
  console.error("Pagefind is not installed. Run npm ci first.");
  process.exit(1);
}
const result = spawnSync(process.execPath, [
  cli, "--site", site, "--output-path", join(out, "pagefind"),
], { stdio: "inherit", timeout: 120_000 });
if (result.error) {
  console.error(`Search indexing failed: ${result.error.message}`);
}
process.exit(result.status ?? 1);
