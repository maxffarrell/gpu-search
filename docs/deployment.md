# Deployment

Public demo: https://gpu-search.vercel.app

GitHub: https://github.com/maxffarrell/gpu-search

Published 2026-09-12 to Vercel project `gpu-search` in `maxffarrells-projects`. The Vercel team API reported billing plan `hobby` before project creation. No paid features or plan changes were enabled. The demo is static; it requires no secrets, server functions, database, analytics, or inference service.

The initial CLI production deployment `dpl_BYWZ9EjyZYaZL8aZKLegvPk3G8K5` reached READY and was aliased to the public URL. Vercel connected the GitHub repository for subsequent builds. `vercel.json` sets the frozen pnpm install, Vite build, output directory and restrictive security headers. `.vercelignore` excludes research data, checkpoints and model assets from the deployment upload.

Run `vercel deploy --prod --scope maxffarrells-projects` to republish when explicitly intended. Runtime behavior was validated on the local production build; deployment readiness and the alias were verified from Vercel's build result.
