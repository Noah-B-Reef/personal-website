// Prefer the project's pinned binary; otherwise use Tola from PATH.
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const local = fileURLToPath(new URL("../.tools/bin/tola", import.meta.url));
const result = spawnSync(existsSync(local) ? local : "tola", process.argv.slice(2), {
  stdio: "inherit",
});
if (result.error) {
  console.error("Tola could not start. Install Tola 0.8.1; see README.md.");
  process.exit(1);
}
process.exit(result.status ?? 1);
